from enum import Enum
from typing import Optional, Tuple, Union, Any
from functools import partial
from dataclasses import dataclass
import matplotlib.pyplot as plt
from dash.exceptions import InvalidConfig
from einops import repeat, rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import PreTrainedModel, GPT2Model, DepthAnythingForDepthEstimation, MambaModel
from mamba_ssm import Mamba
from transformers.utils import ModelOutput
from models.config_dvgformer import DVGFormerConfig
from data.state_action_conversion import state_avg, state_std, action_avg, action_std, reverse_states_actions_tensor
from utils.padding import concated_seq_to_instances, padded_seq_to_instances, padding


class DVGFormerForwardMode(Enum):
    Training = 0
    Generation = 1


class QualityTokenizer(nn.Module):
    '''
    A class to extract quality tokens.
    '''

    def __init__(self, config):
        '''
        Args:
            config (DVGFormerConfig): the configuration for the model
        '''
        super().__init__()
        self.config = config
        self.num_bins = config.num_quantile_bins
        if config.use_quality_mlps:
            self.embed_quality = nn.Sequential(
                nn.Linear(1, config.hidden_size), nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        else:
            self.embed_quality = nn.Embedding(self.num_bins, config.hidden_size)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): [batch_size]
        Returns:
            state (torch.Tensor): [batch_size, hidden_size]
        '''
        if self.config.use_quality_mlps:
            dtype = self.embed_quality[0].weight.dtype
            quality = self.embed_quality(x.unsqueeze(-1).to(dtype) / self.num_bins)
        else:
            quality = self.embed_quality(x)
        return quality


class ImageTokenizer(nn.Module):
    '''
    A class to extract image tokens based on a vision backbone and TokenLearner, following Robotics Transformer.
    '''

    def __init__(self, config):
        '''
        Args:
            config (DVGFormerConfig): the configuration for the model
        '''
        super().__init__()
        self.use_depth = config.use_depth
        self.vision_backbone = config.vision_backbone
        self.image_featmap_shape = config.image_featmap_shape
        self.hidden_size = config.hidden_size

        # backbone
        # dinov2_vits14_reg
        # self.backbone = torch.hub.load(
        #     'facebookresearch/dinov2', config.vision_backbone)
        self.backbone = torch.hub.load('/home/jinpeng-yu/.cache/torch/hub/facebookresearch_dinov2_main',
                                       config.vision_backbone, source='local')
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        # same as LLaVA, two-layer MLP after fixed vision backbone
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(config.image_featmap_shape),
            nn.Conv2d(config.vision_feat_dim, config.hidden_size, 1),
            nn.GELU(),
            nn.Conv2d(config.hidden_size, config.hidden_size, 1),
        )

        if config.use_depth:
            self.depth_model = DepthAnythingForDepthEstimation.from_pretrained(
                'depth-anything/Depth-Anything-V2-Small-hf')
            self.depth_model.eval()
            for parameter in self.depth_model.parameters():
                parameter.requires_grad_(False)
            self.depth_feat = nn.Sequential(
                nn.AdaptiveAvgPool2d(config.image_featmap_shape),
                nn.Conv2d(1, config.hidden_size, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(config.hidden_size, config.hidden_size, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(config.hidden_size, config.hidden_size, 3, 1, 1),
            )
        else:
            self.depth_model = None
            self.depth_feat = None

    def _extract_backbone_features(self, images):
        '''
        Only extract image features from the backbone. No learnable parameters.
        Args:
            images (torch.Tensor): [batch_size, num_channels, height, width]
        Returns:
            feature (torch.Tensor): [batch_size, hidden_size, height, width]
        '''
        B, C, H, W = images.shape
        output = self.backbone.forward_features(images)
        feat = output['x_norm_patchtokens']
        patch_size = self.backbone.patch_size
        h, w = H // patch_size, W // patch_size
        feature = rearrange(feat, 'b (h w) c -> b c h w', h=h, w=w)
        return feature

    def forward(self, images):
        '''
        Args:
            images (torch.Tensor): [batch_size, num_channels, height, width]
            states (torch.Tensor): [batch_size, state_dim]
            intrinsic (torch.Tensor): [batch_size, 3, 3]
        Returns:
            image_tokens (torch.Tensor): [batch_size, n_token_image, hidden_size]
            original_feature (torch.Tensor): [batch_size, hidden_size, height, width]
            disparity (torch.Tensor): [batch_size, height, width]
        '''
        original_feature = self._extract_backbone_features(images)
        feature = self.bottleneck(original_feature)
        if self.use_depth:
            outputs = self.depth_model(images)
            disparity = outputs.predicted_depth
            depth_feature = self.depth_feat(disparity.unsqueeze(1))
            feature = feature + depth_feature
        else:
            disparity = None
        feature = F.adaptive_avg_pool2d(feature, self.image_featmap_shape)
        image_tokens = rearrange(feature, 'b c h w -> b (h w) c')
        return image_tokens


@dataclass
class DVGFormerOutput(ModelOutput):
    """
    A class to store the output of the DVGFormerModel.
    """
    loss: Optional[torch.FloatTensor] = None
    drone_type_preds: Optional[torch.FloatTensor] = None
    action_preds: Optional[torch.FloatTensor] = None
    stop_preds: Optional[torch.FloatTensor] = None
    future_action_preds: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cache_params: Optional[Any] = None


class DVGFormerModel(PreTrainedModel):
    """
    Drone VideoGraphy Transformer Model.
    """

    config_class = DVGFormerConfig
    _supports_flash_attn_2 = True  # dummy value to pass the check in the base class

    def __init__(self, config):
        '''
        Args:
            config (DVGFormerConfig): the configuration for the model
        '''
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_data_frames = config.max_data_frames
        self.max_model_frames = config.max_model_frames
        self.n_future_frames = config.n_future_frames
        self.n_future_steps = config.n_future_frames // config.fps_downsample

        # begin of action token
        self.boa_token_embed = nn.Parameter(torch.randn(config.hidden_size))
        # padding token
        self.pad_token_embed = nn.Parameter(torch.randn(config.hidden_size))

        # number of tokens for describing the entire sequence
        self.n_token_noise = config.n_token_noise
        self.n_token_quality = config.n_token_quality
        self.n_token_drone_type = config.n_token_drone_type
        self.n_token_prepend = config.n_token_prepend
        # fps downsample ratio
        self.fps_downsample = config.fps_downsample
        self.n_action_to_predict = config.n_action_to_predict
        self.per_token_preds = config.per_token_preds
        # number of tokens for describing one frame
        self.n_token_state = config.n_token_state
        self.n_token_image = config.n_token_image
        self.n_token_boa = config.n_token_boa
        self.n_token_action = config.n_token_action
        self.n_token_frame = config.n_token_frame
        # number of tokens to predict
        self.n_token_predict = config.n_token_predict

        # token_types: 0 for predicting nothing, 1 for next state pred, 2 for action pred, 3 for both state and action pred, 4 for stop pred
        # within-frame positional embedding
        self.in_frame_pe = nn.Embedding(config.n_token_frame, config.hidden_size // 2)
        # cross-frame positional embeddings
        self.cross_frame_pe = nn.Embedding(
            config.max_model_frames // config.fps_downsample,
            config.hidden_size // 2)
        self.prepend_pe = nn.Parameter(torch.randn(3, config.hidden_size))

        self.motion_option = config.motion_option

        # tokens for the entire sequence
        self.embed_quality = QualityTokenizer(config)
        self.embed_drone_type = nn.Embedding(2, config.hidden_size)  # 0: non-fpv, 1: fpv
        # tokens for each frame
        self.embed_img = ImageTokenizer(config)
        self.embed_state = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            # nn.LayerNorm(config.hidden_size)
        )
        self.embed_action = nn.Sequential(
            nn.Linear(config.action_dim * config.per_token_preds, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            # nn.LayerNorm(config.hidden_size)
        )
        self.embed_ln = nn.LayerNorm(config.hidden_size)




        if self.config.attention_model == 'GPT2':
            self.transformer = GPT2Model(config.attention_model_config)
            # set the original positional embeddings to zero
            self.transformer.wpe.weight.data.zero_()
            # turn off requires_grad for the original positional embeddings
            self.transformer.wpe.requires_grad_(False)
        elif self.config.attention_model == 'Mamba':
            self.transformer = MambaModel(config.attention_model_config)
            # self.transformer = Mamba(d_model=self.hidden_size)
        else:
            raise InvalidConfig




        # binary classification for drone type, non-fpv vs fpv
        # take image features at t=0
        self.predict_drone_type = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # b, c, h, w -> b, c, 1, 1
            nn.Flatten(start_dim=1),  # b, c, 1, 1 -> b, c
            nn.Linear(config.vision_feat_dim, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, 1),
        )
        # predictions for each frame
        self.predict_action = nn.Linear(config.hidden_size, config.action_dim * config.per_token_preds)
        self.predict_stop = nn.Linear(config.hidden_size, 1)  # binary classification for end-of-seq
        # auxiliary predictions
        self.predict_next_state = nn.Linear(config.hidden_size, config.state_dim * config.per_token_preds)
        self.predict_future_action = nn.Linear(config.hidden_size, config.action_dim * self.n_future_steps)

        self.stop_loss = partial(torchvision.ops.sigmoid_focal_loss, alpha=config.focal_alpha)
        self.drone_type_loss = nn.BCEWithLogitsLoss(reduction='none')

        # Initialize weights and apply final processing
        self.post_init()

    def _tokenize(
        self,
        time_steps: torch.LongTensor,
        images: torch.FloatTensor,
        states: torch.FloatTensor,
        actions: torch.FloatTensor,
        seq_length: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        noise_embed: Optional[torch.LongTensor] = None,
        quality: Optional[torch.LongTensor] = None,
        drone_type: Optional[torch.LongTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ):
        """
        Convert input data to token embeddings and add within/across frame positional embeddings.
        (noise_embed, quality, drone_type, i_1, s_1, a_1, i_2, s_2, a_2, ...)
        Args:
            time_steps (torch.LongTensor): [batch_size, padded_length]
            images (torch.FloatTensor): [batch_size, padded_length, num_channels, height, width]
            states (torch.FloatTensor): [batch_size, padded_length, n_action_to_predict, state_dim]
            actions (torch.FloatTensor): [batch_size, padded_length, n_action_to_predict, action_dim]
            seq_length (torch.LongTensor): [batch_size]
            attention_mask (torch.FloatTensor): [batch_size, padded_length]
            noise_embed (torch.FloatTensor): [batch_size, hidden_size]
            quality (torch.LongTensor): [batch_size]
            drone_type (torch.LongTensor): [batch_size]
            intrinsics (torch.FloatTensor): [batch_size, 3, 3]
            past_key_values (Tuple[Tuple[torch.FloatTensor]]): past key values for the transformer
        Returns:
            stacked_embeds (torch.FloatTensor): [batch_size, stacked_length, hidden_size]
            stacked_attn_mask (torch.FloatTensor): [batch_size, stacked_length]
            position_ids (torch.LongTensor): [batch_size, stacked_length]
            within_frame_pos (torch.LongTensor): [batch_size, stacked_length]
            token_types (torch.LongTensor): [batch_size, stacked_length]
        """
        device = images.device
        b, l = states.shape[:2]
        n = self.n_token_frame

        # embedding for each frame
        _images = padded_seq_to_instances(images, seq_length)
        _images = torch.cat(_images, dim=0)
        _states = padded_seq_to_instances(states, seq_length)
        _states = torch.cat(_states, dim=0)
        _actions = padded_seq_to_instances(actions, seq_length)
        _actions = torch.cat(_actions, dim=0)
        if intrinsics is not None:
            _intrinsics = intrinsics.float()[:, None].repeat(1, l, 1, 1)
            _intrinsics = padded_seq_to_instances(_intrinsics, seq_length)
            _intrinsics = torch.cat(_intrinsics, dim=0)
        else:
            _intrinsics = None

        # image embeddings
        image_embeds = self.embed_img(
            _images.to(self.dtype))
        image_embeds = concated_seq_to_instances(
            image_embeds, seq_length)
        image_embeds = padding(image_embeds, self.config.pad_side,
                               self.config.pad_token_value, l)

        # state embeddings
        state_embeds = self.embed_state(_states.to(self.dtype))
        state_embeds = concated_seq_to_instances(
            state_embeds, seq_length)
        state_embeds = padding(state_embeds, self.config.pad_side,
                               self.config.pad_token_value, l)

        # action embeddings
        action_embeds = self.embed_action(_actions.to(self.dtype))
        action_embeds = concated_seq_to_instances(
            action_embeds, seq_length)
        action_embeds = padding(action_embeds, self.config.pad_side,
                                self.config.pad_token_value, l)

        boa_embeds = self.boa_token_embed.repeat(b, l, 1)
        pred_steps = self.n_action_to_predict // self.per_token_preds

        # attention mask
        stacked_attn_mask = repeat(attention_mask, 'b l -> b (l n)',
                                   n=n).to(self.dtype)

        # stack token embeddings within each time step
        stacked_embeds = torch.cat(
            [state_embeds[:, :, :self.n_token_state],
             image_embeds,
             boa_embeds[:, :, None][:, :, :self.n_token_boa],
             action_embeds],
            dim=2)
        # tokens for t=0, 1, ...
        stacked_embeds = rearrange(stacked_embeds, 'b l n c -> b (l n) c')
        # replace the tokens outside the sequence with padding tokens
        stacked_embeds[~stacked_attn_mask.bool()[:, -l * n:]] = \
            self.pad_token_embed
        # token_types: 0 for predicting nothing, 1 for next state pred, 2 for action pred, 3 for both state and action pred, 4 for stop pred
        token_types = torch.cat(
            [torch.zeros([b, l, self.n_token_state], device=device, dtype=torch.long),
             torch.zeros([b, l, self.n_token_image], device=device,
                         dtype=torch.long),
             torch.ones([b, l, self.n_token_boa], device=self.device,
                        dtype=torch.long) * 2,  # boa
             torch.ones([b, l, pred_steps - 1], device=self.device,
                        dtype=torch.long) * 3,  # action 0~n-2
             torch.ones([b, l, 1], device=self.device,
                        dtype=torch.long) * 1  # last action n-1
             ], dim=2)
        token_types = rearrange(token_types, 'b l n -> b (l n)')

        # across-frame positional embeddings
        position_ids = repeat(time_steps, 'b l -> b (l n)', n=n)
        # within-frame positional embeddings
        within_frame_pos = torch.arange(n, dtype=torch.long, device=device)
        within_frame_pos = repeat(within_frame_pos, 'n -> b (l n)', b=b, l=l)

        # only prepend quality and drone type tokens for input_embeds if needed
        noise_embed = noise_embed.to(device).to(
            self.dtype) if noise_embed is not None else None
        quality_embed = self.embed_quality(
            quality) if quality is not None else None
        drone_type_embed = self.embed_drone_type(
            drone_type) if drone_type is not None else None
        for embed, n_token in zip([noise_embed, quality_embed, drone_type_embed],
                                  [self.n_token_noise, self.n_token_quality, self.n_token_drone_type]):
            if embed is None or n_token == 0:
                continue
            stacked_embeds = torch.cat([embed[:, None][:, :n_token],
                                        stacked_embeds], dim=1)
            position_ids = torch.cat([torch.ones([b, n_token], dtype=torch.long, device=device
                                                 ) * self.config.ignore_value,
                                      position_ids], dim=1)
            within_frame_pos = torch.cat([torch.ones([b, n_token], dtype=torch.long, device=device
                                                     ) * self.config.ignore_value,
                                          within_frame_pos], dim=1)
            # token_types: 0 for predicting nothing, 1 for next state pred, 2 for action pred, 3 for both state and action pred, 4 for stop pred
            token_types = torch.cat([torch.zeros([b, n_token], dtype=torch.long, device=device),
                                     token_types], dim=1)
            # attention mask always include the prepended quality and drone type tokens
            stacked_attn_mask = torch.cat([torch.ones([b, n_token], dtype=self.dtype, device=device),
                                           stacked_attn_mask], dim=1)

        if past_key_values is not None:
            stacked_attn_mask = torch.cat([torch.ones([b, past_key_values[0][0].shape[2]],
                                                      dtype=self.dtype, device=device),
                                           stacked_attn_mask[:, -l * n:]], dim=1)
        return stacked_embeds, stacked_attn_mask, position_ids, within_frame_pos, token_types

    def _init_frame_pred(
        self,
        images: torch.FloatTensor,
    ):
        """
        Predict the drone type based on the image at t=0.
        Args:
            time_steps (torch.LongTensor): [batch_size, padded_length]
            images (torch.FloatTensor): [batch_size, padded_length, num_channels, height, width]
        Returns:
            drone_type_preds (torch.FloatTensor): [batch_size]
        """
        # estimate the drone type
        init_imgs = images[:, 0]
        init_img_feat = self.embed_img._extract_backbone_features(
            init_imgs.to(self.dtype))  # b, l, c
        drone_type_preds = self.predict_drone_type(init_img_feat)  # b, 1
        return drone_type_preds

    def forward(
        self,
        noise_embed: Optional[torch.LongTensor] = None,
        quality: Optional[torch.LongTensor] = None,
        drone_type: Optional[torch.LongTensor] = None,
        intrinsic: Optional[torch.FloatTensor] = None,
        time_steps: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        seq_length: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        within_frame_pos: Optional[torch.LongTensor] = None,
        token_types: Optional[torch.LongTensor] = None,
        drone_type_labels: Optional[torch.FloatTensor] = None,
        next_state_labels: Optional[torch.FloatTensor] = None,
        action_labels: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.FloatTensor] = None,
        future_action_labels: Optional[torch.FloatTensor] = None,
        depth_labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_params = None
    ) -> Union[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], DVGFormerOutput]:
        """
        Forward pass of the model given inputs.
        There are two options for the forward pass:
        1. Training: the model predicts the next token based on inputs of
                     (noise_embed, quality, drone_type, intrinsic, time_steps, images, states, actions, seq_length, attention_mask).
                     Usually not used with past_key_values.
                     To calculate the loss, provide (drone_type_labels, next_state_labels, action_labels, stop_labels, depth_labels).
        2. Generation: the model predicts the next self.n_action_to_predict actions based inputs of
                       (inputs_embeds, position_ids, within_frame_pos, token_types, attention_mask).
                       Usually used together with past_key_values.
                       Loss calculated outside of this function.

        Args:
            noise_embed (torch.LongTensor): [batch_size, hidden_size]
            quality (torch.LongTensor): [batch_size]
            drone_type (torch.LongTensor): [batch_size]
            intrinsic (torch.FloatTensor): [batch_size, 3, 3]
            time_steps (torch.LongTensor): [batch_size, padded_length]
            images (torch.FloatTensor): [batch_size, padded_length, 3, height, width]
            states (torch.FloatTensor): [batch_size, padded_length, n_action_to_predict, state_dim]
            actions (torch.FloatTensor): [batch_size, padded_length, n_action_to_predict, action_dim]
            seq_length (torch.LongTensor): [batch_size]
            past_key_values (Tuple[Tuple[torch.Tensor]]): the past key values for the transformer
            attention_mask (torch.FloatTensor): [batch_size, padded_length] / [batch_size, stacked_length]
            inputs_embeds (torch.FloatTensor): [batch_size, stacked_length, hidden_size]
            position_ids (torch.LongTensor): [batch_size, stacked_length]
            within_frame_pos (torch.LongTensor): [batch_size, stacked_length]
            token_types (torch.LongTensor): [batch_size, stacked_length]
            drone_type_labels (torch.LongTensor): [batch_size]
            next_state_labels (torch.FloatTensor): [batch_size, padded_length, n_action_to_predict, state_dim]
            action_labels (torch.FloatTensor): [batch_size, padded_length, n_action_to_predict, action_dim]
            stop_labels (torch.FloatTensor): [batch_size, padded_length]
            future_action_labels (torch.FloatTensor): [batch_size, padded_length, n_step, action_dim]
            depth_labels (torch.FloatTensor): [batch_size, padded_length, height, width]
            use_cache (bool): whether to use cache for the transformer
            output_hidden_states (bool): whether to output hidden states
            output_attentions (bool): whether to output attentions
        Returns:
            DVGFormerOutput: the output of the model
        """
        # bool indicators
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # L: stacked_length (padded_length * n_token_frame)
        # l: padded_length
        if inputs_embeds is not None:
            device = inputs_embeds.device
            b, L = inputs_embeds.shape[:2]
            l = int(np.ceil(L / self.n_token_frame))
            if L % self.n_token_frame != 0:
                # only predict the last frame (for generation)
                l = min(l, 1)
        else:
            device = states.device
            b, l = states.shape[:2]
            L = l * self.n_token_frame

        if inputs_embeds is None:
            # for training
            # calculate the (inputs_embeds,
            #                position_ids,
            #                within_frame_pos,
            #                token_types,
            #                attention_mask (staked_length))

            # only include the quality and drone type tokens at the beginning of the sequence
            if past_key_values is not None:
                quality = None
                drone_type = None
            # predict drone type
            drone_type_preds = self._init_frame_pred(images)

            inputs_embeds, attention_mask, position_ids, within_frame_pos, token_types = self._tokenize(
                time_steps, images, states, actions, seq_length, attention_mask,
                noise_embed, quality, drone_type, intrinsic, past_key_values)
        else:
            # for generation
            # use the input (inputs_embeds,
            #                position_ids,
            #                within_frame_pos,
            #                token_types,
            #                attention_mask (staked_length))
            assert attention_mask is not None
            assert position_ids is not None
            assert within_frame_pos is not None
            assert token_types is not None

            # if the input is inputs_embeds, drone_type_preds should have been calculated elsewhere
            drone_type_preds = None

        inputs_embeds = self.embed_ln(inputs_embeds)
        # Note: we have kept the positional embeddings to 0 in the original GPT2 model
        # token-level & frame-level positional embedding
        prepend_length = (within_frame_pos == self.config.ignore_value).sum(dim=1)[0].item()
        inputs_embeds[:, :prepend_length] += self.prepend_pe[:prepend_length]
        inputs_embeds[:, prepend_length:] += torch.stack(
            [self.cross_frame_pe(position_ids[:, prepend_length:]),
             self.in_frame_pe(within_frame_pos[:, prepend_length:])], dim=-1).flatten(-2)
        # sanity check
        inputs_embeds = inputs_embeds.to(self.dtype)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        if self.config.attention_model == 'GPT2':
            past_key_values = None if past_key_values is None else tuple(
                tuple(pkv.to(self.dtype) for pkv in pkvs) for pkvs in past_key_values)

            transformer_outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=torch.zeros_like(position_ids),
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
            hidden_states = transformer_outputs[0]
        elif self.config.attention_model == 'Mamba':
            transformer_outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                # attention_mask=attention_mask,
                # use_cache=True,
                # cache_params=None,
                # cache_position = torch.tensor([position_ids[position_ids != -100][0].item()] * 4,
                #                               dtype=torch.long, device=device),
                output_hidden_states=output_hidden_states
            )
            hidden_states = transformer_outputs.last_hidden_state
            # transformer_outputs = self.transformer(hidden_states=inputs_embeds)
            # hidden_states = transformer_outputs
        else:
            raise InvalidConfig

        # get predictions
        # input:    image * n_token_image, (state, action) * n_action_to_predict
        # predict:  (action, next_state) * n_action_to_predict, stop

        # token_types: 0 for predicting nothing, 1 for next state pred, 2 for action pred, 3 for both state and action pred, 4 for stop pred
        # predict next state
        pred_next_state_pos = (token_types == 1) | (token_types == 3)
        if pred_next_state_pos.any():
            next_state_preds = self.predict_next_state(hidden_states[pred_next_state_pos])
            next_state_preds = next_state_preds.view([b, l, -1, self.config.state_dim])
        else:
            next_state_preds = None
        # predict next action
        pred_action_pos = (token_types == 2) | (token_types == 3)
        if pred_action_pos.any():
            action_preds = self.predict_action(hidden_states[pred_action_pos])
            action_preds = action_preds.view([b, l, -1, self.config.action_dim])
        else:
            action_preds = None
        # predict begin-of-frame / end-of-sequence
        # first state to predict action
        pred_stop_pos = ((token_types == 2) | (token_types == 3)) & (
            torch.cat([torch.zeros([b, 1], dtype=torch.bool, device=device),
                       token_types[:, :-1] == 0], dim=1))
        if pred_stop_pos.any():
            stop_preds = self.predict_stop(hidden_states[pred_stop_pos])
            stop_preds = rearrange(stop_preds, '(b l) 1 -> b l', b=b)
            future_action_preds = self.predict_future_action(hidden_states[pred_stop_pos])
            future_action_preds = rearrange(future_action_preds, '(b l) (n c) -> b l n c',
                                            b=b, n=self.n_future_steps, c=self.config.action_dim)
        else:
            stop_preds = None
            future_action_preds = None

        loss = None
        if stop_labels is not None:
            # drone type
            if drone_type_preds is not None and drone_type_labels is not None:
                init_idx = time_steps[:, 0] == 0
                loss_drone_type = (self.drone_type_loss(
                    drone_type_preds[:, 0], drone_type_labels.to(self.dtype)) * init_idx).mean()
            else:
                loss_drone_type = 0
            # sequence
            loss_next_state = F.l1_loss(next_state_preds, next_state_labels.to(self.dtype),
                                        reduction='none').mean(dim=[2, 3]) if next_state_preds is not None else 0
            loss_action = F.l1_loss(action_preds, action_labels.to(self.dtype),
                                    reduction='none').mean(dim=[2, 3]) if action_preds is not None else 0
            loss_stop = self.stop_loss(stop_preds, stop_labels.to(self.dtype),
                                       reduction='none')
            future_action_mask = (future_action_labels !=
                                  self.config.ignore_value).all(dim=-1, keepdim=True)
            loss_future_action = ((future_action_preds - future_action_labels.to(self.dtype)).abs() *
                                  future_action_mask).mean(dim=[2, 3]) / self.fps_downsample

            sequence_mask = stop_labels != self.config.ignore_value
            seq_loss = (loss_next_state * self.config.loss_coef_state +
                        loss_action * self.config.loss_coef_action +
                        loss_stop * self.config.loss_coef_stop +
                        loss_future_action * self.config.loss_coef_future) * sequence_mask
            loss = loss_drone_type * self.config.loss_coef_drone_type + seq_loss.mean()

        if self.config.attention_model == 'GPT2':
            return DVGFormerOutput(
                loss=loss,
                drone_type_preds=drone_type_preds,
                action_preds=action_preds,
                stop_preds=stop_preds,
                future_action_preds=future_action_preds,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions
            )
        elif self.config.attention_model == 'Mamba':
            return DVGFormerOutput(
                loss=loss,
                drone_type_preds=drone_type_preds,
                action_preds=action_preds,
                stop_preds=stop_preds,
                future_action_preds=future_action_preds,
                hidden_states=hidden_states,
                cache_params=None
            )
        else:
            raise InvalidConfig

    def _reverse_states_actions(self, states: torch.FloatTensor, actions: torch.FloatTensor):
        """
        Reverse the states and actions to get the next state.
        Args:
            states (torch.FloatTensor): [batch_size, n_action_to_predict, state_dim]
            actions (torch.FloatTensor): [batch_size, n_action_to_predict, action_dim]
        Returns:
            next_states (torch.FloatTensor): [batch_size, n_action_to_predict, state_dim]
            (vs, omegas) (Tuple[torch.FloatTensor, torch.FloatTensor]): [batch_size, n_action_to_predict, 3]
        """

        device = states.device

        # l: padded_length for images & states
        b, l = states.shape[:2]

        # TODO: this is way too slow
        # get next state based on the predicted action and its conversion
        # unnormalize the state: tvec and qvec (global)
        _state_avg = torch.tensor(state_avg, dtype=torch.float32).to(device)
        _state_std = torch.tensor(state_std, dtype=torch.float32).to(device)
        _action_avg = torch.tensor(action_avg, dtype=torch.float32).to(device)
        _action_std = torch.tensor(action_std, dtype=torch.float32).to(device)
        states_unnorm = states.float() * _state_std + _state_avg
        # unnormalize the action: v and omega (local)
        actions_unnorm = actions.float() * _action_std + _action_avg
        # reverse state and actions
        # tvec(t+1), qvec(t+1), v(t), omega(t)
        next_tvecs, next_qvecs, vs, omegas = reverse_states_actions_tensor(
            states_unnorm.flatten(0, 2), actions_unnorm.flatten(0, 2),
            motion_option=self.motion_option)
        _next_states = torch.cat([next_tvecs, next_qvecs], dim=-1)
        # normalize the next states
        next_states = (_next_states - _state_avg) / _state_std
        next_states = rearrange(next_states, '(b l n) c -> b l n c',
                                b=b, l=l, c=self.config.state_dim)
        return next_states, (vs, omegas)

    def validate(
        self,
        conv_state,
        ssm_state,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        within_frame_pos: Optional[torch.LongTensor] = None,
        token_types: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_params=None
    ): # -> Union[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], DVGFormerOutput]:
        # bool indicators
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # L: stacked_length (padded_length * n_token_frame)
        # l: padded_length
        device = inputs_embeds.device
        b, L = inputs_embeds.shape[:2]
        # l = int(np.ceil(L / self.n_token_frame))
        # if L % self.n_token_frame != 0:
        #     # only predict the last frame (for generation)
        #     l = min(l, 1)

        assert attention_mask is not None
        assert position_ids is not None
        assert within_frame_pos is not None
        assert token_types is not None

        # if the input is inputs_embeds, drone_type_preds should have been calculated elsewhere
        drone_type_preds = None

        inputs_embeds = self.embed_ln(inputs_embeds)
        # Note: we have kept the positional embeddings to 0 in the original GPT2 model
        # token-level & frame-level positional embedding
        prepend_length = (within_frame_pos == self.config.ignore_value).sum(dim=1)[0].item()
        inputs_embeds[:, :prepend_length] += self.prepend_pe[:prepend_length]
        inputs_embeds[:, prepend_length:] += torch.stack(
            [self.cross_frame_pe(position_ids[:, prepend_length:]),
             self.in_frame_pe(within_frame_pos[:, prepend_length:])], dim=-1).flatten(-2)

        # sanity check
        inputs_embeds = inputs_embeds.to(self.dtype)
        # if self.past_input is None:
        #     self.past_input = inputs_embeds
        # else:
        #     self.past_input = torch.cat([self.past_input, inputs_embeds], dim=1)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        if self.config.attention_model == 'GPT2':
            past_key_values = None if past_key_values is None else tuple(
                tuple(pkv.to(self.dtype) for pkv in pkvs) for pkvs in past_key_values)

            transformer_outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=torch.zeros_like(position_ids),
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
            hidden_states = transformer_outputs[0]
        elif self.config.attention_model == 'Mamba':
            cache_position = torch.tensor([4 if inputs_embeds.shape[1] == 1 else 0], dtype=torch.long, device=device)

            transformer_outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                cache_params=cache_params,
                cache_position=cache_position,
                output_hidden_states=output_hidden_states
            )
            hidden_states = transformer_outputs.last_hidden_state
            # transformer_outputs, conv_state, ssm_state = self.transformer.step(
            #     hidden_states=inputs_embeds,
            #     conv_state=conv_state,
            #     ssm_state=ssm_state
            # )
            # hidden_states = transformer_outputs
        else:
            raise InvalidConfig

        # get predictions
        # input:    image * n_token_image, (state, action) * n_action_to_predict
        # predict:  (action, next_state) * n_action_to_predict, stop

        # token_types: 0 for predicting nothing, 1 for next state pred, 2 for action pred, 3 for both state and action pred, 4 for stop pred
        # predict next action
        # if self.past_input.shape[1] != token_types.shape[1]:
        #     token_types_padding = torch.zeros((1, self.past_input.shape[1] - token_types.shape[1]), dtype=torch.bool, device=device)
        #     token_types = torch.cat((token_types_padding, token_types), dim=1)

        pred_action_pos = (token_types == 2) | (token_types == 3)
        if pred_action_pos.any():
            action_preds = self.predict_action(hidden_states)
            # action_preds = self.predict_action(hidden_states[pred_action_pos])
            action_preds = action_preds.view([b, 1, -1, self.config.action_dim])
        else:
            action_preds = None

        # predict begin-of-frame / end-of-sequence
        # first state to predict action
        stop_preds = None
        future_action_preds = None

        if self.config.attention_model == 'GPT2':
            return DVGFormerOutput(
                loss=None,
                drone_type_preds=drone_type_preds,
                action_preds=action_preds,
                stop_preds=stop_preds,
                future_action_preds=future_action_preds,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions
            )
        elif self.config.attention_model == 'Mamba':
            return DVGFormerOutput(
                loss=None,
                drone_type_preds=drone_type_preds,
                action_preds=action_preds,
                stop_preds=stop_preds,
                future_action_preds=future_action_preds,
                hidden_states=hidden_states,
                cache_params=transformer_outputs.cache_params
            ), conv_state, ssm_state
        else:
            raise InvalidConfig

    def expand_actions(
        self,
        conv_state,
        ssm_state,
        noise_embed: Optional[torch.LongTensor] = None,
        quality: Optional[torch.LongTensor] = None,
        drone_type: Optional[torch.LongTensor] = None,
        intrinsic: Optional[torch.FloatTensor] = None,
        time_steps: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        seq_length: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cache_params = None,
    ): # -> DVGFormerOutput:
        """
        Expand the actions for one (current) pair of image & state (for generation).
        There are three modes for expanding the action based on config.test_gt_forcing:
        1. 'allframe': use GT actions in each frame (15 fps) as conditions for expanding the actions. must be used when actions are provided.
        2. 'keyframe': use GT state for key frames (3 fps) and PREDICTED actions (15 fps) as conditions for expanding the actions
        3. 'none': same as above, but also execute the actions to get the future image & state
                   (require support for interactive environment)

        Args:
            quality (torch.LongTensor): [batch_size]
            drone_type (torch.LongTensor): [batch_size]
            intrinsic (torch.FloatTensor): [batch_size, 3, 3]
            time_steps (torch.LongTensor): [batch_size, padded_length]
            images (torch.FloatTensor): [batch_size, padded_length, num_channels, height, width]
            states (torch.FloatTensor): [batch_size, padded_length, n_action_to_predict, state_dim]
            actions (torch.FloatTensor): [batch_size, padded_length, n_action_to_predict, action_dim]
            seq_length (torch.LongTensor): [batch_size]
            past_key_values (Tuple[Tuple[torch.Tensor]]): the past key values for the transformer
            attention_mask (torch.FloatTensor): [batch_size, padded_length]
        Returns:
            DVGFormerOutput: the output of the model
        """
        device = images.device

        # l: padded_length for images & states
        b, l = states.shape[:2]
        # only compute embeddings for the last frame
        l = 1
        # past_key_values: list of config.n_layer elements,
        # each a two-tuple of key, value of shape [batch_size, num_head, seq_length, C]
        # remove previous frames as they should be included in the past_key_values
        time_steps = time_steps[:, -l:]
        images = images[:, -l:]
        states = executed_states = states[:, -l:]
        actions = actions[:, -l:]
        seq_length = torch.ones([b], dtype=torch.long) * l

        # only include quality and drone_type for the first frame
        if past_key_values is None:
            drone_type_preds = self._init_frame_pred(images)
            drone_type = (drone_type_preds[:, 0] > 0).type(torch.long)
        else:
            quality = None
            drone_type = None
            drone_type_preds = None
            noise_embed = None

        inputs_embeds, attention_mask, position_ids, within_frame_pos, token_types = self._tokenize(
            time_steps, images, states, actions, seq_length, attention_mask.to(self.dtype),
            noise_embed, quality, drone_type, intrinsic, past_key_values)
        # remove the last several terms for prediction
        if self.n_token_predict == 0:
            end_idx = inputs_embeds.shape[1]
            mask_end_idx = attention_mask.shape[1]
        else:
            end_idx = mask_end_idx = -self.n_token_predict
        inputs_embeds = inputs_embeds[:, :end_idx]

        # from transformers.generation.utils import GenerationMixin
        # input:    quality, drone_type, image * n_token_image, (state, action) * n_action_to_predict
        # predict:  (state, action) * n_action_to_predict, stop
        model_inputs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask[:, :mask_end_idx],
            'position_ids': position_ids[:, :end_idx],
            'within_frame_pos': within_frame_pos[:, :end_idx],
            'token_types': token_types[:, :end_idx],
            'past_key_values': past_key_values,
            # 'use_cache': True,
            'cache_params': cache_params
        }

        # predict the next self.n_action_to_predict actions & next_states
        # t=0,1,2,...
        action_preds = torch.zeros([b, 1, self.n_action_to_predict, self.config.action_dim],
                                   device=device, dtype=self.dtype)
        # t=1,2,3,...
        next_states = torch.zeros([b, 1, self.n_action_to_predict, self.config.state_dim],
                                  device=device, dtype=self.dtype)
        pred_steps = self.n_action_to_predict // self.per_token_preds
        gt_forcing = self.config.test_gt_forcing == 'allframe' and not (
            actions == self.config.ignore_value).any().item() and not (
            states == self.config.ignore_value).any().item()

        for i in range(self.n_token_predict + 1):
            outputs, conv_state, ssm_state = self.validate(conv_state, ssm_state, **model_inputs)

            model_inputs['past_key_values'] = outputs.past_key_values
            model_inputs['cache_params'] = outputs.cache_params
            if outputs.action_preds is not None:
                # action
                if self.per_token_preds == 1:
                    action_preds[:, :, i] = outputs.action_preds[:, :, 0]
                else:
                    action_preds = outputs.action_preds
                # which action to use as condition for the next generation iteration
                # see for the three modes of config.test_gt_forcing
                # t=0,1,2,3,4
                if i < pred_steps:
                    if gt_forcing:
                        executed_actions = actions
                    else:
                        executed_actions = action_preds
                # next state
                if self.per_token_preds == 1:
                    next_states[:, :, i] = self._reverse_states_actions(
                        executed_states, executed_actions)[0][:, :, i]
                else:
                    for ii in range(self.per_token_preds):
                        next_states[:, :, i + ii] = self._reverse_states_actions(
                            executed_states, executed_actions)[0][:, :, i + ii]
                # t=1,2,3,4
                if i + 1 < pred_steps:
                    if gt_forcing:
                        executed_states[:, :, i + 1] = states[:, :, i + 1]
                    else:
                        executed_states[:, :, i + 1] = next_states[:, :, i]

            # stop
            stop_preds = outputs.stop_preds
            future_action_preds = outputs.future_action_preds
            # if outputs.stop_preds is not None:
            #     stop_preds = outputs.stop_preds
            # # future action
            # if outputs.future_action_preds is not None:
            #     future_action_preds = outputs.future_action_preds

            # next token
            if self.n_token_action == 1 and i < pred_steps:
                next_embeds = self.embed_action(executed_actions.to(self.dtype)[:, -1, [i]])
            if i == self.n_token_predict:
                break

            # update the input kwargs for next generation iteration
            seq_idx = -self.n_token_predict + i
            inputs_embeds = torch.cat([inputs_embeds, next_embeds], dim=1)
            # [b, 1]
            model_inputs['inputs_embeds'] = next_embeds
            model_inputs['position_ids'] = position_ids[:, [seq_idx]]
            model_inputs['within_frame_pos'] = within_frame_pos[:, [seq_idx]]
            model_inputs['token_types'] = token_types[:, [seq_idx]]
            # [b, seq_length]
            attn_end_idx = seq_idx + 1
            model_inputs['attention_mask'] = (attention_mask[:, :attn_end_idx]
                                              if attn_end_idx < 0 else attention_mask)

        return DVGFormerOutput(
            drone_type_preds=drone_type_preds,
            action_preds=action_preds,
            stop_preds=stop_preds,
            future_action_preds=future_action_preds,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cache_params=outputs.cache_params
        ), conv_state, ssm_state


def main():
    pass


if __name__ == '__main__':
    main()
