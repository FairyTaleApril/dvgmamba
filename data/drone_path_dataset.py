import os
import re
import time
import logging
import json
from typing import Dict, Sequence
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torchvision.transforms as T
import torchvision.transforms.functional as F

from configs.config_drone_path_dataset import DronePathDatasetConfig
from data.state_action_conversion import get_states_actions, reverse_states_actions, state_avg, state_std, action_avg, action_std
from utils.padding import padding
from utils.quaternion_operations import convert_to_local_frame, horizontal_flip, add_angular_velocity_to_quaternion, quaternions_to_angular_velocity
from utils.flexible_fs import FlexibleFileSystem
from utils.colmap_official_read_write_model import read_cameras_binary

fpv_keywords = ['fpv', 'dji avata']
drone_keywords = ['drone', 'dji', 'fpv']
skip_words = ['unbox', 'piyush', 'drone catches', 'glue gun', 'humne', 'gets attacked', 'drone attack', 'sonic drone',
              'zz kid', 'merekam', 'scary video', 'drone show', 'Arcade Craniacs', '.exe', 'Stromedy', 'zombie',
              'restore', 'Jester', 'drone strike', 'Mark Rober', 'Plasmonix', 'Sourav Joshi Vlogs', 'lego', 'MR. INDIAN HACKER',
              'horror', 'ketakutan', 'Mikael TubeHD', 'TRT \u00c7ocuk', 'kurulacak', 'penampakan', 'The RawKnee Games',
              'NIkku Vlogz', 'girgaya', 'chinu', 'military', 'surveillance', 'pakai', 'Yudist Ardhana', 'Technical Indian with fun',
              'menangkap', 'nampak', 'ZZ Kids TV', 'soldier', 'Bharat Tak', 'Sannu Kumar', 'lahari', 'Mikael Family', 'Smart rohit gadget',
              'attacked by', 'Josh Reid', 'terekam', 'kentang', 'sumur maut', 'Frost Diamond', 'Fatih Can Aytan', 'apex legends',
              'Golden TV - Funny Hindi Comedy Videos', 'hindi comedy', 'The Squeezed Lemon', 'no one would have believed it',
              'Wonders of the World', 'carwow', 'no one would believe it', 'Max TV', 'cartoon', 'for kids', 'pendaki', 'andrea ramadhan',
              'tornado', 'corpse', 'aerial tale', 'muthumalai', 'Videogyan Kids Shows - Toddler Learning Videos', 'WasimOP', 'nangis',
              'Zefanya Oyanio', 'deadly fire', 'oblivion', 'darussalam', 'Kang Adink', 'shoot down', 'collapse', 'hamas', 'aftermath',
              'NewsFirst Kannada', 'drone fails', 'FailArmy', 'YaQurban', 'pashto', 'camping', 'Harmen Hoek', 'slaughter', 'weapon',
              'JerryRigEverything', 'toys', 'Majedar Review', ]


def get_video_stat(metadata):
    def contain(xs, keywords):
        return any(keyword in x for keyword in keywords for x in xs)

    like_count = metadata.get('like_count', None)
    comment_count = metadata.get('comment_count', None)
    is_fpv = contain([
        metadata['title'].lower(),
        metadata['description'].lower()
    ], fpv_keywords)
    is_drone = contain([
        metadata['title'].lower(),
        metadata['description'].lower()
    ], drone_keywords)
    has_skip_words = contain([
        metadata['title'].lower(),
        metadata['description'].lower(),
        metadata.get('channel', '')
    ], skip_words)
    width, height = map(lambda x: int(x), metadata['resolution'].split('x'))
    is_landscape = width / height > 1

    video_stat = {
        'title': metadata['title'],
        'view_count': metadata['view_count'],
        'like_count': like_count if like_count is not None else 0,
        'comment_count': comment_count if comment_count is not None else 0,
        'duration': metadata['duration'],
        'is_fpv': is_fpv,
        'has_skip_words': has_skip_words,
        'is_drone': is_drone,
        'is_landscape': is_landscape,
        # 'has_skip_language': has_skip_language(metadata['title'].lower()),
    }
    return video_stat


def color_jitter(img, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, fn_idx=(0, 1, 2, 3)):
    for fn_id in fn_idx:
        if fn_id == 0 and brightness:
            img = F.adjust_brightness(img, 1 + brightness)
        elif fn_id == 1 and contrast:
            img = F.adjust_contrast(img, 1 + contrast)
        elif fn_id == 2 and saturation:
            img = F.adjust_saturation(img, 1 + saturation)
        elif fn_id == 3 and hue:
            img = F.adjust_hue(img, hue)
    return img


def get_noise_vector(index, noise_dim=384):
    random_generator = np.random.RandomState(index)
    return random_generator.randn(noise_dim)


class DronePathDataset(Dataset):

    def __init__(self, config: DronePathDatasetConfig):
        super().__init__()

        self.root = config.root
        self.hdf5_fname = config.hdf5_fname

        self.noise_dim = config.hidden_size

        self.max_data_frames = config.max_data_frames
        self.max_model_frames = config.max_model_frames

        self.original_fps = config.original_fps
        self.resolution = config.resolution
        self.fix_image_width = config.fix_image_width

        self.drone_types = config.drone_types
        self.motion_option = config.motion_option
        self.fps_downsample = config.fps_downsample
        self.action_fps = config.action_fps

        self.num_quantile_bins = config.num_quantile_bins
        self.ignore_value = config.ignore_value

        # fewer images if original_fps > fps: only one image every fps_downsample frames
        # (image, state, action), (state, action), ..., (state, action)
        self.action_downsample = config.action_downsample
        self.n_action_to_predict = config.n_action_to_predict
        # future prediction
        self.n_future_steps = config.n_future_frames // self.fps_downsample  # at fps

        # consider vertical videos (portrait mode)
        self.skip_portrait_videos = config.skip_portrait_videos

        # augmentation
        self.random_horizontal_flip = config.random_horizontal_flip
        self.random_scaling = config.random_scaling
        self.random_temporal_crop = config.random_temporal_crop
        self.random_color_jitter = config.random_color_jitter

        self.transform_img = T.Compose([
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Update: change from tar to HDF5 for the SWMR mode for multi workers in dataloader
        self.h5_fs = FlexibleFileSystem(f'{self.root}/{self.hdf5_fname if self.hdf5_fname else ""}')
        self.video_stats = {}
        self.all_scenes = []
        self.data_list = []
        self.points_info = []
        total_length = 0

        for video_id in tqdm(sorted(self.h5_fs.listdir(self.root))):
            try:
                with self.h5_fs.open(f'{self.root}/{video_id}/data.json') as f:
                    metadata = json.load(f)
                video_stat = get_video_stat(metadata)
                video_stat['num_clips'] = 0
                video_stat['quality'] = (video_stat['view_count'] / video_stat['duration'])

                if not video_stat['is_drone'] or video_stat['has_skip_words']:
                    continue
                if self.skip_portrait_videos and not video_stat['is_landscape']:
                    continue
                if int(video_stat['is_fpv']) not in self.drone_types:
                    continue

                self.video_stats[video_id] = video_stat
            except:
                # print(f'Error reading metadata for {video_id}: {e}')
                continue

            current_num_clips = len(self.data_list)
            for result_fname in sorted(self.h5_fs.listdir(f'{self.root}/{video_id}')):
                if not ('-score' in result_fname and result_fname.endswith('.csv')):
                    continue

                scene = os.path.basename(result_fname).split('-')[0].replace('scene', '')
                video_scene = f'{video_id}/scene{scene}'
                if video_scene not in self.all_scenes:
                    self.all_scenes.append(video_scene)

                recons_index = int(os.path.basename(result_fname).split('-')[1].replace('recons', ''))
                score = int(re.search(r'-score(\d+)', result_fname).group(1))
                valid = '_invalid' not in result_fname
                if not score or not valid:
                    continue

                frame_folder = f'{self.root}/{video_id}/scene{scene}-recons{recons_index}-frames/'
                if not self.h5_fs.exists(frame_folder):
                    continue
                result_fpath = f'{self.root}/{video_id}/{result_fname}'

                with self.h5_fs.open(result_fpath, 'r') as f:
                    recons_df = pd.read_csv(f, comment='#')
                recons_array = recons_df.to_numpy()

                total_length += len(recons_array) / self.original_fps
                if self.max_model_frames == self.max_data_frames:
                    self.data_list.append({
                        'result_fpath': result_fpath,
                        'start_idx': 0,
                        'end_idx': len(recons_array)
                    })
                else:
                    chunk_step = self.max_model_frames // 2 // self.fps_downsample * self.fps_downsample
                    for start_idx in range(0,
                            len(recons_array) // self.fps_downsample * self.fps_downsample - chunk_step, chunk_step):
                        self.data_list.append({
                            'result_fpath': result_fpath,
                            'start_idx': start_idx,
                            'end_idx': min(start_idx + self.max_model_frames, len(recons_array))
                        })
            self.video_stats[video_id]['num_clips'] = len(self.data_list) - current_num_clips

        # log the overall stats
        stat_keys = ['view_count', 'like_count', 'comment_count', 'duration', 'is_fpv', 'num_clips', 'quality']
        video_stat_array = {}
        for key in stat_keys:
            video_stat_array[key] = np.array([self.video_stats[video_id][key] for video_id in self.video_stats])
        is_fpv = video_stat_array["is_fpv"] == 1
        print(f'Dataset: {len(self.data_list)} sequences, {total_length / 3600:.1f} hours')
        print(f'total videos: {len(self.video_stats)} \t'
              f'fpv: {np.sum(is_fpv)} \tnon-fpv: {np.sum(~is_fpv)}')
        print(f'view count: {np.mean(video_stat_array["view_count"]):.1f} \t'
              f'fpv: {np.mean(video_stat_array["view_count"][is_fpv]):.1f} \t'
              f'non-fpv: {np.mean(video_stat_array["view_count"][~is_fpv]):.1f}')

        # quantize the video stats based ont the number of clips
        clip_quality = []
        for video_id in self.video_stats:
            clip_quality.extend([self.video_stats[video_id]['quality']] * self.video_stats[video_id]['num_clips'])
        quantile_bins = np.quantile(clip_quality, np.linspace(0, 1, self.num_quantile_bins))
        for video_id in self.video_stats:
            self.video_stats[video_id]['quality_quantile'] = int(np.digitize(
                self.video_stats[video_id]['quality'], quantile_bins, right=True))

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index, visualize=False):
        t0 = time.time()

        # info
        result_fpath = self.data_list[index]['result_fpath']
        start_idx = self.data_list[index]['start_idx']
        end_idx = self.data_list[index]['end_idx']

        video_id = result_fpath.split('/')[-2]
        video_stats = self.video_stats[video_id]
        scene = os.path.basename(result_fpath).split('-')[0].replace('scene', '')
        recons_index = int(os.path.basename(result_fpath).split('-')[1].replace('recons', ''))
        is_fpv = int(video_stats['is_fpv'])

        # recons info
        with self.h5_fs.open(result_fpath, 'r') as f:
            coord_multiplier = float(f.readline().decode().replace('#', '').strip())
            recons_df = pd.read_csv(f, comment='#')

        # augmentation for entire sequence
        H, W = self.resolution
        # random horizontal flip
        if self.random_horizontal_flip:
            flip = np.random.rand() < 0.5
        else:
            flip = False
        noise_idx = int(flip)
        # random scaling
        if self.random_scaling:
            # random scale up, i.e., no padding, only cropping
            scale = np.random.uniform(1.0, 1.2)
            H = int(H * scale)
            W = int(W * scale)
        else:
            scale = 1.0
        # random temporal crop
        if self.max_data_frames == self.max_model_frames and np.random.rand() < self.random_temporal_crop / 2:
            # reduce the sequence length by start_offset, end_offset
            seq_length = end_idx - start_idx
            start_offset = np.random.randint(0, seq_length * 0.2 + 1)
            start_offset = start_offset // self.fps_downsample * self.fps_downsample
            start_idx += start_offset
            # end_offset = np.random.randint(
            #     0, seq_length * 0.2 + 1)
            # end_offset = end_offset // self.fps_downsample * self.fps_downsample
            # end_idx -= end_offset
        # random color jitter
        if self.random_color_jitter:
            brightness = np.random.uniform(-0.5, 0.5)
            contrast = np.random.uniform(-0.5, 0.5)
            saturation = np.random.uniform(-0.5, 0.5)
            hue = 0.0
            jitter_fn_idx = np.random.permutation(4)
        else:
            brightness = 0.0
            contrast = 0.0
            saturation = 0.0
            hue = 0.0
            jitter_fn_idx = np.arange(4)

        noise_embed = get_noise_vector(noise_idx + index * 2, self.noise_dim)

        # camera poses
        # note: the original tvecs and qvecs in read_images_binary() gives camera extrinsic matrix [R=quat2mat(qvec), t=tvec],
        # but the camera pose (location and orientation) in the global coord system is [-R.T @ t, R.T]
        # recons_df include the converted tvecs and qvecs (all in world coord system)
        recons_array = recons_df.to_numpy()[start_idx:]
        # camera path in global coord system (measurements)
        # raw_tvecs = recons_array[:, 1:4].astype(float)
        # raw_qvecs = recons_array[:, 4:8].astype(float)
        # raw_vs = recons_array[:, 8:11].astype(float)
        # raw_omegas = recons_array[:, 11:14].astype(float)
        # camera path the global coord system (estimation)
        raw_tvecs = recons_array[:, 14:17].astype(float)
        raw_qvecs = recons_array[:, 17:21].astype(float)
        raw_vs = recons_array[:, 21:24].astype(float)
        raw_omegas = recons_array[:, 24:27].astype(float)
        # add the final speed and angular velocity to extend the sequence
        final_tvec = raw_tvecs[-1] + raw_vs[-1]
        final_qvec = add_angular_velocity_to_quaternion(raw_qvecs[-1], raw_omegas[-1], 1)
        raw_tvecs = np.concatenate([raw_tvecs, final_tvec[None]], axis=0)
        raw_qvecs = np.concatenate([raw_qvecs, final_qvec[None]], axis=0)

        # reference frame
        ref_tvec, ref_qvec = raw_tvecs[0], raw_qvecs[0]

        # change the global coord system to the initial frame
        tvecs = np.zeros_like(raw_tvecs)
        qvecs = np.zeros_like(raw_qvecs)
        for i in range(len(raw_tvecs)):
            tvecs[i], qvecs[i], _, _ = convert_to_local_frame(ref_tvec, ref_qvec, raw_tvecs[i], raw_qvecs[i])

        # modulate the speed based on the drone type
        # tvecs *= self.drone_speeds[is_fpv]
        # vs *= self.drone_speeds[is_fpv]
        # sequence length based on the start and end index
        seq_length = (end_idx - start_idx) // self.fps_downsample
        # time steps (same length as image/state)
        time_steps = np.arange(seq_length)
        # augmentation
        if flip:
            aug_tvecs = np.zeros_like(tvecs)
            aug_qvecs = np.zeros_like(qvecs)
            for i in range(len(tvecs)):
                aug_tvecs[i], aug_qvecs[i], _, _ = horizontal_flip(tvecs[i], qvecs[i])
        else:
            aug_tvecs, aug_qvecs = tvecs, qvecs

        # states & actions are self.n_action_to_predict times the length
        time_range = (time_steps[:, None] * self.fps_downsample + np.arange(self.n_action_to_predict))
        _states, _actions = get_states_actions(aug_tvecs, aug_qvecs,
            motion_option=self.motion_option, action_downsample=self.action_downsample)
        # include the last state
        next_t, next_q, _, _ = reverse_states_actions(_states[[-1]], _actions[[-1]], motion_option=self.motion_option)
        _states = np.concatenate([_states, np.concatenate([next_t, next_q], axis=1)], axis=0)
        next_states = _states[time_range // self.action_downsample + 1]
        states = _states[time_range // self.action_downsample]
        # if sparse action, make sure the action is still of the same norm
        actions = (_actions[time_range // self.action_downsample] / self.action_downsample)

        # compute the future waypoints
        n_step_vs = np.ones([seq_length + self.n_future_steps - 1, 3]) * self.ignore_value
        n_step_omegas = np.ones([seq_length + self.n_future_steps - 1, 3]) * self.ignore_value
        for i in range(seq_length + self.n_future_steps - 1):
            if (time_steps[0] + i + 1) * self.fps_downsample >= len(aug_tvecs):
                break
            n_step_vs[i] = (aug_tvecs[(time_steps[0] + i + 1) * self.fps_downsample] -
                            aug_tvecs[(time_steps[0] + i) * self.fps_downsample])
            n_step_omegas[i] = quaternions_to_angular_velocity(
                aug_qvecs[(time_steps[0] + i) * self.fps_downsample],
                aug_qvecs[(time_steps[0] + i + 1) * self.fps_downsample], 1)
            if self.motion_option == 'local':
                # change the global coord system to the initial frame
                _, _, n_step_vs[i], n_step_omegas[i] = convert_to_local_frame(
                    aug_tvecs[(time_steps[0] + i) * self.fps_downsample],
                    aug_qvecs[(time_steps[0] + i) * self.fps_downsample],
                    None, None, n_step_vs[i], n_step_omegas[i])
        n_step_actions = np.concatenate(
            [n_step_vs, n_step_omegas], axis=1)[np.arange(seq_length)[:, None] + np.arange(self.n_future_steps)]

        t1 = time.time()

        # load images
        frame_folder = f'{self.root}/{video_id}/scene{scene}-recons{recons_index}-frames'
        image_dict = {}
        for i in (time_steps * self.fps_downsample).tolist():
            fname = recons_array[i, 0]
            if self.h5_fs.h5_file is not None:
                with self.h5_fs.open(f'{frame_folder}/{fname}', 'r') as f:
                    img = Image.open(f).convert('RGB')
            else:
                img = Image.open(f'{frame_folder}/{fname}').convert('RGB')
            image_dict[fname] = img

        if visualize:
            imgs = [self.transform_img(img) for img in image_dict.values()]
            T.ToPILImage()(make_grid(torch.stack(imgs), normalize=True)).save('frames.jpg')

        t2 = time.time()

        # camera intrinsics
        cameras_info = read_cameras_binary(
            f'{self.root}/{video_id}/scene{scene}-recons{recons_index}-colmap/cameras.bin',
            self.h5_fs
        )
        cam = cameras_info[1]
        # original camera parameters
        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = cam.params[0]
            cx = cam.params[1]
            cy = cam.params[2]
        elif cam.model in (
            "PINHOLE",
            "OPENCV",
            "OPENCV_FISHEYE",
            "FULL_OPENCV",
        ):
            fx = cam.params[0]
            fy = cam.params[1]
            cx = cam.params[2]
            cy = cam.params[3]
        else:
            raise Exception("Camera model not supported")

        # resize the image to match the resolution
        h, w = cam.height, cam.width
        # assert cx * 2 == w and cy * 2 == h, 'principal point should be at the center'
        if self.fix_image_width and not self.skip_portrait_videos:
            # resize the image to match the width
            img_ratio = W / w
        else:
            # resize the image to match the smaller dimension, i.e., no padding, only cropping
            img_ratio = max(W / w, H / h)
        target_h = int(img_ratio * h)
        target_w = int(img_ratio * w)
        # also update the intrinsics for the resized image
        fx, fy = fx * img_ratio, fy * img_ratio
        # cx, cy = cx * img_ratio, cy * img_ratio
        # will center crop the image so the principal point should always be at the center
        cx, cy = W / 2, H / 2
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        # images
        images = []
        for i in (time_steps * self.fps_downsample).tolist():
            fname = recons_array[i, 0]
            img = image_dict[fname].convert('RGB')
            # PIL in w, h format
            img = img.resize((target_w, target_h))
            # flip
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # color jitter
            if self.random_color_jitter:
                img = color_jitter(img, brightness, contrast, saturation, hue, jitter_fn_idx)
            img = self.transform_img(img)
            images.append(img)
        images = torch.stack(images)

        t3 = time.time()

        time_steps = torch.tensor(time_steps, dtype=torch.long)
        states = (states - state_avg) / state_std
        states = torch.tensor(states, dtype=torch.float32)
        next_states = (next_states - state_avg) / state_std
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = (actions - action_avg) / action_std
        actions = torch.tensor(actions, dtype=torch.float32)
        stop_labels = torch.zeros(seq_length)
        future_mask = (n_step_actions != self.ignore_value).all(axis=-1)
        n_step_actions[future_mask] = (n_step_actions[future_mask] - action_avg) / action_std
        n_step_actions = torch.tensor(n_step_actions, dtype=torch.float32)

        scene_part = int(scene.split('_')[1])
        next_scene = f'{scene.split("_")[0]}_{scene_part + 1}'
        is_partial_scene = f'{video_id}/scene{next_scene}' in self.all_scenes
        if end_idx >= len(recons_df) // self.fps_downsample * self.fps_downsample and not is_partial_scene:
            stop_labels[-1] = 1

        data_dict = {
            # inputs
            'noise_embed': torch.tensor(noise_embed, dtype=torch.float32),
            'quality': video_stats['quality_quantile'],
            'drone_type': is_fpv,
            'intrinsic': torch.from_numpy(K).float(),
            'time_steps': time_steps,
            'images': images,
            'states': states,
            'actions': actions,
            'seq_length': seq_length,
            # labels
            'next_state_labels': next_states,
            'action_labels': actions.clone(),
            'stop_labels': stop_labels,
            'future_action_labels': n_step_actions,
            'drone_type_labels': is_fpv,
        }

        t4 = time.time()
        logging.debug(f'read csv: {t1 - t0:.3f}s, \tffmpeg: {t2 - t1:.3f}s, \t'
                      f'conversion: {t3 - t2:.3f}s, \toutput: {t4 - t3:.3f}s, \t'
                      f'total: {t4 - t0:.3f}s')
        return data_dict

    def __del__(self):
        if self.h5_fs is not None:
            self.h5_fs.close()


def collate_fn_video_drone_path_dataset(
    instances: Sequence[Dict],
    pad_side='right', pad_value=0, label_pad_value=-1000
) -> Dict[str, torch.Tensor]:
    if 'noise_embed' not in instances[0]:
        batch = {
            'images': padding([instance['images'] for instance in instances], pad_side, pad_value),
            'seq_length': torch.tensor([instance['seq_length'] for instance in instances]),
            'states': padding([instance['states'] for instance in instances], pad_side, pad_value),
            'actions': padding([instance['actions'] for instance in instances], pad_side, pad_value),
            'action_labels': padding([instance['action_labels'] for instance in instances], pad_side, label_pad_value),
        }
        return batch


def main():
    from train import get_args_dict

    config = DronePathDatasetConfig(get_args_dict())
    dataset = DronePathDataset(config)
    print(len(dataset))

    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i, visualize=True)


if __name__ == '__main__':
    main()
