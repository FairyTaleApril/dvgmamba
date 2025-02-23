from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MixerModel
from mamba_ssm.utils.generation import InferenceParams
from transformers.utils import ModelOutput

from configs.config_dvgmamba import DVGMambaConfig
# from models.mixer_seq_simple import MixerModel
from models.tokenizer import FrameTokenizer


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

        self.n_action_to_predict = config.n_action_to_predict
        self.loss_coef_action = config.loss_coef_action

        self.embedding = FrameTokenizer(config=config)

        self.transformer = MixerModel(
            d_model = self.hidden_size,
            n_layer = self.n_layer,
            d_intermediate = self.hidden_size * 2,
            residual_in_fp32=True,
            device = config.device,
            dtype = config.dtype
        )

        self.predict_action = nn.Linear(self.hidden_size, self.action_dim)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        images: Optional[torch.Tensor],
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> DVGMambaOutput:
        """
        Args:
            images (torch.Tensor): [batch_size, padded_length, 3, height, width]
            states (torch.Tensor): [batch_size, padded_length, n_action_to_predict, state_dim]
            actions (torch.Tensor): [batch_size, padded_length, n_action_to_predict, action_dim]
            action_labels (torch.Tensor): [batch_size, padded_length, n_action_to_predict, action_dim]
            cache (torch.Tensor)
        Returns:
            DVGMambaOutput: the output of the model
        """
        images = images.to(device=self.device, dtype=self.dtype)
        if states is not None:
            states = states.to(device=self.device, dtype=self.dtype)
        if actions is not None:
            actions = actions.to(device=self.device, dtype=self.dtype)
        if action_labels is not None:
            action_labels = action_labels.to(device=self.device, dtype=self.dtype)
        if cache is not None:
            cache = cache.to(device=self.device)

        if self.training:
            input_embeddings, token_types = self.embedding(images)
            hidden_states = self.transformer(
                hidden_states=input_embeddings,
                inference_params=None
            )

            hidden_action_preds = hidden_states[:, -self.n_action_to_predict:, :]
            action_preds = self.predict_action(hidden_action_preds).unsqueeze(0)
            # action_preds = action_preds.view([b, l, -1, self.action_dim])

            loss_action = F.l1_loss(action_preds, action_labels[:, -1, :, :].unsqueeze(1).to(self.dtype), reduction='none').mean(dim=[2, 3])
            loss = (loss_action * self.loss_coef_action).mean()

            return DVGMambaOutput(
                loss=loss,
                action_preds=action_preds,
                cache=None
            )
        else:
            b, cross_pos_ids = 1, cache.shape[1] // 46 if cache is not None else 0
            inference_params = InferenceParams(max_seqlen=100, max_batch_size=100)
            action_preds = torch.zeros([1, self.n_action_to_predict, self.action_dim], device=self.device, dtype=self.dtype)

            input_embeddings, _ = self.embedding(images, cross_pos_ids=cross_pos_ids, see=True)
            all_input_embeddings = torch.cat([cache, input_embeddings], dim=1) if cache is not None else input_embeddings

            hidden_states = self.transformer(
                hidden_states=all_input_embeddings,
                # inference_params=inference_params
                inference_params=None
            )
            # inference_params.seqlen_offset += all_input_embeddings.shape[1]
            # action_preds[:, 0, :] += self.predict_action(hidden_states[:, -1, :].unsqueeze(1))[:, 0]
            action_preds[:, :, :] += self.predict_action(hidden_states[:, -self.n_action_to_predict:, :])

            if all_input_embeddings.shape[1] >= 1000:
                pass

            # input_embeddings, _ = self.embedding(
            #     images,
            #     actions=action_preds[:, 0, :].unsqueeze(1).unsqueeze(1),
            # )
            # input_embeddings = input_embeddings[:, -1, :].unsqueeze(1)
            # all_input_embeddings = torch.cat([all_input_embeddings, input_embeddings], dim=1)
            #
            # for i in range(self.n_action_to_predict - 1):
            #     hidden_states = self.transformer(
            #         hidden_states=input_embeddings,
            #         inference_params=inference_params
            #     )
            #     inference_params.seqlen_offset += 1
            #     action_preds[:, i+1, :] += self.predict_action(hidden_states)[:, 0]
            #
            #     input_embeddings, _ = self.embedding(
            #         images,
            #         actions=action_preds[:, :i+2, :].unsqueeze(1),
            #     )
            #     input_embeddings = input_embeddings[:, -1, :].unsqueeze(1)
            #     all_input_embeddings = torch.cat([all_input_embeddings, input_embeddings], dim=1)

            return DVGMambaOutput(
                loss=None,
                action_preds=action_preds.unsqueeze(1),
                cache=all_input_embeddings
            )

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")
        return total_params

    def load_model(self, checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.to(device=self.device, dtype=self.dtype)
