from configs.base_config import BaseConfig


class DronePathDatasetConfig(BaseConfig):

    def __init__(self, args_dict):
        super().__init__()

        self.root = args_dict['root']
        self.hdf5_fname = args_dict['hdf5_fname']

        self.n_future_frames = args_dict.get('n_future_frames', 15)

        self.skip_portrait_videos = args_dict.get('skip_portrait_videos', True)

        self.random_horizontal_flip = args_dict.get('random_horizontal_flip', False)
        self.random_scaling = args_dict.get('random_scaling', False)
        self.random_temporal_crop = args_dict.get('random_temporal_crop', False)
        self.random_color_jitter = args_dict.get('random_color_jitter', False)
