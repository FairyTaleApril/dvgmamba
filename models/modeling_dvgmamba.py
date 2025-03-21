from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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

        if self.training:
            b, l = images.shape[:2]

            input_embeddings, token_types = self.embedding(
                images=images,
                states=states,
                actions=actions
            )

            hidden_states = self.mamba(
                input_ids=input_embeddings,
                inference_params=None
            )

            pred_action_pos = (token_types == 2) | (token_types == 3)
            action_preds = self.predict_action(hidden_states[pred_action_pos])
            action_preds = action_preds.view([b, l, -1, self.action_dim])

            mask = torch.arange(l, device=self.device).unsqueeze(0) < seq_length.to(device=self.device, dtype=self.dtype).unsqueeze(1)  # [b, l]
            loss_action = F.l1_loss(action_preds, action_labels.to(device=self.device, dtype=self.dtype), reduction='none')
            loss_action = loss_action.mean(dim=[2, 3])  # [b, l]
            loss_action = (loss_action * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = (loss_action * self.loss_coef_action).mean()

            return DVGMambaOutput(
                loss=loss,
                action_preds=action_preds,
                cache=None
            )
        else:
            b, l = 1, 1
            cross_pos_ids, inference_params = cache['cross_pos_ids'], cache['inference_params']

            states = rearrange(states, 'n d -> 1 1 n d')
            action_preds = torch.zeros([b, l, self.n_action_to_predict, self.action_dim], device=self.device, dtype=self.dtype)

            use_cache = True
            if not use_cache:
                for i in range(self.n_action_to_predict):
                    all_input_embeddings, _ = self.embedding(
                        images=images,
                        states=states,
                        actions=action_preds[..., :i, :] if i != 0 else None,
                        cross_pos_ids=cross_pos_ids,
                        past_input_embeddings=cache['all_input_embeddings'],
                    )

                    hidden_states = self.mamba(
                        input_ids=all_input_embeddings,
                        inference_params=None
                    )

                    action_preds[..., i, :] += self.predict_action(hidden_states[..., -1:, :])
            else:
                input_embeddings_cache, _ = self.embedding(
                    images=images,
                    states=states,
                    actions=None,
                    cross_pos_ids=cross_pos_ids,
                )

                hidden_states = None
                if cache['all_input_embeddings'] is not None:
                    hidden_states = self.mamba(
                        input_ids=cache['all_input_embeddings'][..., -1:, :],
                        inference_params=inference_params
                    )
                    inference_params.seqlen_offset += 1
                for i in range(input_embeddings_cache.shape[1]):
                    hidden_states = self.mamba(
                        input_ids=input_embeddings_cache[..., i:i + 1, :],
                        inference_params=inference_params
                    )
                    inference_params.seqlen_offset += 1

                action_preds[..., 0, :] += self.predict_action(hidden_states)

            # # self.see_hidden = see_params(self.see_hidden, hidden_states[..., -self.n_action_to_predict:, :], 'b n d -> b n d')
            # self.see_action = see_params(self.see_action, action_preds, 'b l n d -> b (l n) d')
            #
            # if cross_pos_ids >= 29:
            #     self.count += 1
            #     if self.count == 12:
            #         pass

            all_input_embeddings, _ = self.embedding(
                images=images,
                states=states,
                actions=action_preds,
                cross_pos_ids=cross_pos_ids,
                past_input_embeddings=cache['all_input_embeddings'],
            )

            return DVGMambaOutput(
                loss=None,
                action_preds=action_preds,
                cache={
                    'cross_pos_ids': cross_pos_ids,
                    'all_input_embeddings': all_input_embeddings,
                    'inference_params': inference_params,
                }
            )

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")
        return total_params

    def load_model(self, checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.to(device=self.device, dtype=self.dtype)
