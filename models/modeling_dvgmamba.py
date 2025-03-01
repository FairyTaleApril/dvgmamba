from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MixerModel
from transformers.utils import ModelOutput

from configs.config_dvgmamba import DVGMambaConfig
from models.tokenizer import FrameTokenizer, see_params


@dataclass
class DVGMambaOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    action_preds: Optional[torch.Tensor] = None
    cache: Optional[Any] = None


class DVGMambaModel(nn.Module):
    """
    Drone VideoGraphy Mamba Model.
    """

    def __init__(self, config: DVGMambaConfig):
        """
        Args:
            config (DVGMambaConfig): the configuration for the model
        """

        super().__init__()

        # self.config = config
        self.device = config.device
        self.dtype = config.dtype

        self.n_layer = config.n_layer
        self.hidden_size = config.hidden_size
        self.action_dim = config.action_dim
        self.max_model_frames = config.max_model_frames

        self.n_action_to_predict = config.n_action_to_predict
        self.loss_coef_action = config.loss_coef_action

        self.embedding = FrameTokenizer(config=config)

        self.mamba = MixerModel(
            d_model = self.hidden_size,
            n_layer = self.n_layer,
            d_intermediate = self.hidden_size * 2,
            vocab_size = 0,
            rms_norm=True,
            residual_in_fp32=True,
            device = config.device,
            dtype = config.dtype
        )
        self.mamba.embedding = nn.Identity()

        self.predict_action = nn.Linear(self.hidden_size, self.action_dim)

        self.see_hidden = None
        self.see_action = None

    def forward(
        self,
        images: Optional[torch.Tensor],
        seq_length: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> DVGMambaOutput:
        """
        Args:
            images (torch.Tensor): [batch_size, padded_length, 3, height, width]
            seq_length (torch.Tensor): [batch_size]
            states (torch.Tensor): [batch_size, padded_length, n_action_to_predict, state_dim]
            actions (torch.Tensor): [batch_size, padded_length, n_action_to_predict, action_dim]
            action_labels (torch.Tensor): [batch_size, padded_length, n_action_to_predict, action_dim]
            cache (torch.Tensor)
        Returns:
            DVGMambaOutput: the output of the model
        """
        images = images.to(device=self.device, dtype=self.dtype)
        if seq_length is not None:
            seq_length = seq_length.to(device=self.device, dtype=self.dtype)
        if states is not None:
            states = states.to(device=self.device, dtype=self.dtype)
        if actions is not None:
            actions = actions.to(device=self.device, dtype=self.dtype)
        if action_labels is not None:
            action_labels = action_labels.to(device=self.device, dtype=self.dtype)

        if self.training:
            b, l = images.shape[:2]

            input_embeddings, token_types = self.embedding(
                images=images
            )

            hidden_states = self.mamba(
                input_ids=input_embeddings,
                inference_params=None
            )

            pred_action_pos = (token_types == 2) | (token_types == 3)
            action_preds = self.predict_action(hidden_states[pred_action_pos])
            action_preds = action_preds.view([b, l, -1, self.action_dim])

            mask = torch.arange(l, device=self.device).unsqueeze(0) < seq_length.unsqueeze(1)  # [b, l]
            loss_action = F.l1_loss(action_preds, action_labels, reduction='none').mean(dim=[2, 3])  # [b, l]
            loss_action = (loss_action * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = (loss_action * self.loss_coef_action).mean()

            return DVGMambaOutput(
                loss=loss,
                action_preds=action_preds,
                cache=None
            )
        else:
            b, l, cross_pos_ids = 1, 1, cache['cross_pos_ids']
            action_preds = torch.zeros([b, l, self.n_action_to_predict, self.action_dim], device=self.device, dtype=self.dtype)
            if cache['all_input_embeddings'] is not None:
                cache['all_input_embeddings'] = cache['all_input_embeddings'].to(device=self.device, dtype=self.dtype)

            input_embeddings, _ = self.embedding(
                images=images,
                cross_pos_ids=cross_pos_ids,
                see=True
            )
            all_input_embeddings = torch.cat([
                cache['all_input_embeddings'],
                input_embeddings
            ], dim=1) if cache['all_input_embeddings'] is not None else input_embeddings

            hidden_states = self.mamba(
                input_ids=all_input_embeddings,
                inference_params=None
            )

            action_preds += self.predict_action(hidden_states[..., -self.n_action_to_predict:, :])

            if True:
                self.see_hidden = see_params(self.see_hidden, hidden_states[..., -self.n_action_to_predict:, :], 'b n d -> b n d')
                self.see_action = see_params(self.see_action, action_preds, 'b l n d -> b (l n) d')

            if cache['cross_pos_ids'] >= 28:
                pass

            return DVGMambaOutput(
                loss=None,
                action_preds=action_preds,
                cache={'cross_pos_ids': cross_pos_ids, 'all_input_embeddings': all_input_embeddings},
            )

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")
        return total_params

    def load_model(self, checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.to(device=self.device, dtype=self.dtype)
