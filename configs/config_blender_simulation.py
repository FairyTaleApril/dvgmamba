from configs.base_config import BaseConfig


class BlenderSimulationConfig(BaseConfig):

    def __init__(self, logdir, run_name, args_dict):
        super().__init__()

        self.logdir = logdir
        self.run_name = run_name

        self.num_runs = args_dict.get('num_runs', 40)
        self.num_repeats = args_dict.get('num_repeats', 3)
        self.random_init_pose = args_dict.get('random_init_pose', True)
        self.video_duration = args_dict.get('video_duration', 10)
        self.re_render = args_dict.get('re_render', True)

        self.fps = 3  # MUST BE 3
        self.drone_type = 1  # MUST BE 1

        sensor_width = 36.0  # 35mm sensor width
        self.cropped_sensor_width = sensor_width if self.fix_image_width else sensor_width / 16 * 9
