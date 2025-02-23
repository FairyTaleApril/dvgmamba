import json


class BaseConfig:
    base_config = {
        "hidden_size": 384,
        'state_dim': 7,
        'action_dim': 6,

        'max_data_frames': 150,  # max sequence length in data, 10 seconds at 15 fps
        'max_model_frames': 150,  # max sequence length to consider

        'original_fps': 15,  # Original frame rate used for colmap reconstruction
        'resolution': [168, 294],
        'fix_image_width': True,

        'drone_types': [1],
        'motion_option': 'local',  # local or global motion
        # The dataset only has frame_(1 + 5n), so fps downsample MUST BE 5.
        'fps_downsample': 5,
        # The frame rate used for colmap reconstruction is 15 which means the dataset's fps is 3.
        # So if action_fps = 15, the model will predicts 5 actions for 1 image.
        # If action_fps = 3, the model will predicts 1 actions for 1 image.
        'action_fps': 3,

        'num_quantile_bins': 10,
        'ignore_value': -100,
    }

    def __init__(self):
        self.base_config['action_downsample'] = self.base_config['original_fps'] // self.base_config['action_fps']
        self.base_config['n_action_to_predict'] = self.fps_downsample // self.action_downsample

        assert self.max_model_frames % self.fps_downsample == 0, 'max_model_frames should be divisible by fps_downsample'

    @property
    def action_downsample(self):
        return self.base_config['action_downsample']

    @property
    def n_action_to_predict(self):
        return self.base_config['n_action_to_predict']

    @classmethod
    def update_config(cls, **kwargs):
        cls.base_config.update(kwargs)

    def __getattr__(self, name):
        if name in self.base_config:
            return self.base_config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.base_config:
            self.update_config(**{name: value})
        else:
            super().__setattr__(name, value)

    def print_config(self, need_print=True):
        config_dict = {}
        for key, value in vars(self).items():
            if 'device' in key or 'dtype' in key:
                config_dict[key] = str(value)
            else:
                config_dict[key] = value

        config_dict["base_config"] = self.base_config

        config_string = json.dumps(config_dict, indent=4)
        if need_print:
            print(config_string)
        return config_string, config_dict


if __name__ == '__main__':
    from train import get_args_dict
    from configs.config_dvgmamba import DVGMambaConfig
    from configs.config_drone_path_dataset import DronePathDatasetConfig
    from configs.config_blender_simulation import BlenderSimulationConfig

    args_dict = get_args_dict()

    dvgmamba_config = DVGMambaConfig(args_dict)
    dataset_config = DronePathDatasetConfig(args_dict)
    blender_config = BlenderSimulationConfig(logdir='logdir', run_name='run_name', args_dict=args_dict)

    print(dvgmamba_config.print_config())
    print(dataset_config.print_config())
    print(blender_config.print_config())

    dvgmamba_config.hidden_size = 1
    dvgmamba_config.n_layer = 1

    print(dvgmamba_config.print_config())
    print(dataset_config.print_config())
    print(blender_config.print_config())
