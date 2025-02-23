import numpy as np
import torch
from transformers import PretrainedConfig

from configs.base_config import BaseConfig


dtype_map = {
    'torch.float32': torch.float32,
    'torch.float16': torch.float16,
    'torch.bfloat16': torch.bfloat16
}


class DVGMambaConfig(BaseConfig):

    model_type = 'dvgmamba'

    def __init__(self, args_dict):
        super().__init__()

        self.device = torch.device(args_dict.get('device', 'cuda'))
        self.dtype = dtype_map.get(args_dict.get('dtype', 'torch.bfloat16'))
        self.n_layer = args_dict.get('n_layer', 12)
        self.use_depth = args_dict.get('use_depth', False)

        self.vision_backbone = args_dict.get('vision_backbone', 'dinov2_vits14_reg')
        if 'efficientnet' in self.vision_backbone:
            patch_size = 1
            self.backbone_downsample = 32
            self.vision_feat_dim = 1536
            vision_backbone_resolution = 240
        elif 'mobilenet' in self.vision_backbone:
            patch_size = 1
            self.backbone_downsample = 32
            self.vision_feat_dim = 960
            vision_backbone_resolution = 240
        elif 'dinov2' in self.vision_backbone:
            patch_size = 14
            self.backbone_downsample = patch_size
            self.vision_feat_dim = 384
            vision_backbone_resolution = 224
        else:
            raise ValueError(f'Unsupported architecture "{self.vision_backbone}".')

        if self.fix_image_width:
            # keep everything at h:w=9:16
            # same area as [vision_backbone_resolution, vision_backbone_resolution], but with 16:9 aspect ratio
            height = round(vision_backbone_resolution / (16 * 9) ** 0.5 * 9 / patch_size) * patch_size
            width = round(vision_backbone_resolution / (16 * 9) ** 0.5 * 16 / patch_size) * patch_size
            resolution = [height, width]
        else:
            # use aspect ratio of h:w=1:1
            resolution = [vision_backbone_resolution, vision_backbone_resolution]
        self.resolution = resolution

        # TODO: implement this
        self.prediction_option = args_dict.get('prediction_option', 'iterative')
        assert self.prediction_option in ['iterative', 'one-shot'], 'Unknown prediction option: {self.prediction_option}'

        self.image_featmap_shape = (5, 9) if self.fix_image_width else (7, 7)
        if self.image_featmap_shape is not None:
            if isinstance(self.image_featmap_shape, int):
                h, w = self.resolution
                # match to the longer side
                if h < w:
                    image_featmap_shape = (int(np.round(self.image_featmap_shape * h / w)), self.image_featmap_shape)
                else:
                    image_featmap_shape = (self.image_featmap_shape, int(np.round(self.image_featmap_shape * w / h)))
                self.image_featmap_shape = image_featmap_shape
        else:
            self.image_featmap_shape = list(map(
                lambda x: int(np.ceil(np.ceil(x / self.backbone_downsample))),
                self.resolution
            ))

        # tokens for describing one frame
        # image token
        self.n_token_image = int(np.prod(self.image_featmap_shape))
        # state token
        self.n_token_state = args_dict.get('n_token_state', 1)
        # begin of action token
        self.n_token_boa = args_dict.get('n_token_boa', 1)
        # action token
        self.n_token_action = args_dict.get('n_token_action', 1)

        # loss weight
        self.loss_coef_state = args_dict.get('loss_coef_state', 0)
        self.loss_coef_action = args_dict.get('loss_coef_action', 1)
        self.loss_coef_stop = args_dict.get('loss_coef_stop', 0)
