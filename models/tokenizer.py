from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from configs.config_dvgmamba import DVGMambaConfig


def see_params(all_params, new_params, pattern):
    temp = rearrange(new_params, pattern).norm(dim=2, p=2, keepdim=True)
    cat_params = torch.cat([
        all_params,
        temp
    ], dim=1) if all_params is not None else temp
    return cat_params


class ImageTokenizer(nn.Module):
    """
    A class to extract image tokens based on a vision backbone and TokenLearner, following Robotics Transformer.
    """

    def __init__(self, config: DVGMambaConfig):
        """
        Args:
            config (DVGMambaConfig): the configuration for the model
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.use_depth = config.use_depth
        self.image_featmap_shape = config.image_featmap_shape
        self.vision_backbone = config.vision_backbone
        self.vision_feat_dim = config.vision_feat_dim

        # backbone: dinov2_vits14_reg
        self.backbone = torch.hub.load('facebookresearch/dinov2', self.vision_backbone)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        # same as LLaVA, two-layer MLP after fixed vision backbone
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.image_featmap_shape),
            nn.Conv2d(self.vision_feat_dim, self.hidden_size, 1),
            nn.GELU(),
            nn.Conv2d(self.hidden_size, self.hidden_size, 1),
        )

        if self.use_depth:
            from transformers import DepthAnythingForDepthEstimation

            self.depth_model = DepthAnythingForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
            self.depth_model.eval()
            for parameter in self.depth_model.parameters():
                parameter.requires_grad_(False)

            self.depth_feat = nn.Sequential(
                nn.AdaptiveAvgPool2d(self.image_featmap_shape),
                nn.Conv2d(1, self.hidden_size, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(self.hidden_size, self.hidden_size, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(self.hidden_size, self.hidden_size, 3, 1, 1),
            )

    def extract_backbone_features(self, images):
        """
        Only extract image features from the backbone. No learnable parameters.
        Args:
            images (torch.Tensor): [batch_size, num_channels, height, width]
        Returns:
            feature (torch.Tensor): [batch_size, hidden_size, height, width]
        """
        B, C, H, W = images.shape
        h, w = H // self.backbone.patch_size, W // self.backbone.patch_size

        output = self.backbone.forward_features(images)
        feature = rearrange(output['x_norm_patchtokens'], 'b (h w) c -> b c h w', h=h, w=w)
        return feature

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): [batch_size, num_channels, height, width]
        Returns:
            image_tokens (torch.Tensor): [batch_size, n_token_image, hidden_size]
        """
        original_feature = self.extract_backbone_features(images)
        feature = self.bottleneck(original_feature)

        if self.use_depth:
            outputs = self.depth_model(images)
            disparity = outputs.predicted_depth
            depth_feature = self.depth_feat(disparity.unsqueeze(1))
            feature = feature + depth_feature

        feature = F.adaptive_avg_pool2d(feature, self.image_featmap_shape)
        image_tokens = rearrange(feature, 'b c h w -> b (h w) c')
        return image_tokens


class FrameTokenizer(nn.Module):

    def __init__(self, config: DVGMambaConfig):
        """
        Args:
            config (DVGMambaConfig): the configuration for the model
        """
        super().__init__()

        self.device = config.device
        self.dtype = config.dtype

        self.hidden_size = config.hidden_size
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim

        self.max_model_frames = config.max_model_frames
        self.n_token_frame = config.n_token_frame
        self.n_token_image = config.n_token_image
        self.n_token_boa = config.n_token_boa
        self.n_token_state = config.n_token_state
        self.n_action_to_predict = config.n_action_to_predict

        # tokens for each frame
        self.img_embedding = ImageTokenizer(config)
        self.boa_embedding = nn.Parameter(torch.randn(self.hidden_size))
        self.state_embedding = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.LayerNorm(self.hidden_size)
        )
        self.action_embedding = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.LayerNorm(self.hidden_size)
        )

        # self.in_frame_pe = nn.Embedding(self.n_token_frame, self.hidden_size)
        self.cross_frame_pe = nn.Embedding(self.max_model_frames, self.hidden_size)

        self.see_boa = None
        self.see_pe = None
        self.see_images = None
        self.see_image_embeddings = None
        self.see_input_embeddings = None

    def __call__(
        self,
        images: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        cross_pos_ids: Optional[int] = None,
        past_input_embeddings: Optional[torch.Tensor] = None,
        see=False
    ):
        """
        Convert input data to token embeddings and add within/cross frame positional embeddings.
        (s_1, img_1, boa, a_1, s_2, img_2, boa, a_2, ...)
        Args:
            images (torch.Tensor): [batch_size, padded_length, num_channels, height, width]
            states (torch.Tensor): [batch_size, padded_length, n_action_to_predict, state_dim]
            actions (torch.Tensor): [batch_size, padded_length, n_action_to_predict, action_dim]
        Returns:
            input_embeddings (torch.Tensor): [batch_size, stacked_length, hidden_size]
            token_types (torch.Tensor): [batch_size, stacked_length]
        """
        device = images.device
        b, l = images.shape[:2]
        assert l <= self.max_model_frames, 'Sequence length exceeds max_seqlen'
        assert images.shape[1] % self.n_token_image != 0

        n_token_frame = ((self.n_token_state if states is not None else 0)
                         + self.n_token_image
                         + self.n_token_boa
                         + (actions.shape[2] if actions is not None else 0))
        # within_frame_pos = torch.arange(n_token_frame, dtype=torch.long, device=device)
        # within_frame_pos = repeat(within_frame_pos, 'n -> b (l n)', b=b, l=l)
        cross_pos_ids = torch.arange(l, dtype=torch.long, device=device) if cross_pos_ids is None \
            else torch.tensor([cross_pos_ids], dtype=torch.long, device=device)
        cross_frame_pos = repeat(cross_pos_ids, 'l -> b (l n)', b=b, n=n_token_frame)
        # frame_pe = self.in_frame_pe(within_frame_pos) + self.cross_frame_pe(cross_frame_pos)
        frame_pe = self.cross_frame_pe(cross_frame_pos)

        rearranged_images = rearrange(images, "b l c h w -> (b l) c h w")
        image_embeddings = self.img_embedding(rearranged_images)  # (b l) n d
        image_embeddings = rearrange(image_embeddings, "(b l) n d -> b l n d", b=b, l=l)

        state_embeddings = self.state_embedding(states.to(device=self.device, dtype=self.dtype)) if states is not None else None
        boa_embeddings = repeat(self.boa_embedding, 'd -> b l n d', b=b, l=l, n=self.n_token_boa)
        action_embeddings = self.action_embedding(actions.to(device=self.device, dtype=self.dtype)) if actions is not None else None

        embeddings_to_concat = [state_embeddings, image_embeddings, boa_embeddings, action_embeddings]
        valid_embeddings = [embeddings for embeddings in embeddings_to_concat if embeddings is not None]
        input_embeddings = torch.cat(valid_embeddings, dim=2)
        input_embeddings = rearrange(input_embeddings, 'b l n d -> b (l n) d')
        input_embeddings = input_embeddings + frame_pe

        if past_input_embeddings is not None:
            input_embeddings = torch.cat([
                past_input_embeddings.to(device=self.device, dtype=self.dtype),
                input_embeddings,
            ], dim=1)

        tokens_to_concat = [
            torch.ones([b, l, self.n_token_image], device=device, dtype=torch.long) * 0,
            torch.ones([b, l, self.n_token_boa], device=device, dtype=torch.long) * 2  # boa
        ]
        if states is not None:
            state_tokens = [torch.ones([b, l, self.n_token_state], device=device, dtype=torch.long) * 0]
            tokens_to_concat = state_tokens + tokens_to_concat
        if actions is not None:
            if actions.shape[2] == 1:
                action_tokens = [torch.ones([b, l, 1], device=device, dtype=torch.long) * 0]
            else:
                action_tokens = [
                    torch.ones([b, l, actions.shape[2] - 1], device=device, dtype=torch.long) * 2,
                    torch.ones([b, l, 1], device=device, dtype=torch.long) * 0  # last action
                ]
            tokens_to_concat = tokens_to_concat + action_tokens
        token_types = torch.cat(tokens_to_concat, dim=2)
        token_types = rearrange(token_types, 'b l n -> b (l n)')

        if see:
            self.see_images = see_params(self.see_images, images, 'b l c h w -> b l (c h w)')
            self.see_image_embeddings = see_params(self.see_image_embeddings, image_embeddings, 'b l n d -> b (l n) d')
            self.see_boa = see_params(self.see_boa, boa_embeddings, 'b l 1 d -> b l d')
            self.see_input_embeddings = see_params(self.see_input_embeddings, input_embeddings, 'b n d -> b n d')
            self.see_pe = see_params(self.see_pe, frame_pe, 'b n d -> b n d')

        return input_embeddings, token_types
