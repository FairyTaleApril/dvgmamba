import os

import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from PIL import Image
from mamba_ssm.utils.generation import InferenceParams

from blender.blender_camera_env import BlenderCameraEnv
from configs.config_blender_simulation import BlenderSimulationConfig
from data.state_action_conversion import action_avg, action_std, state_avg, state_std
from models.modeling_dvgmamba import DVGMambaModel
from utils.quaternion_operations import convert_to_local_frame

infinigen_root = '/workspace/DVG/infinigen'
blosm_root = '/workspace/DVG/blosm'


def expand_episode(env, model, run_name, seed, random_init_pose, config: BlenderSimulationConfig):
    # Reset environment
    env.drone_type = config.drone_type
    observation, info = env.reset(seed=seed, random_init_pose=random_init_pose)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    seq_len = 1
    done = False
    crash = None
    total_reward = 0
    t_ref, q_ref = np.zeros(3), np.array([1, 0, 0, 0])

    model.eval()
    cache = {
        'cross_pos_ids': 0,
        'all_input_embeddings': None,
        'inference_params': InferenceParams(max_seqlen=config.max_model_frames, max_batch_size=1)
    }

    while not done:
        # Convert observation to tensor & normalize
        image = Image.fromarray(observation['image']).convert('RGB')
        image = transform(image)[None, None]
        state = (observation['state'] - state_avg) / state_std
        states = torch.stack(
            [torch.tensor(state, dtype=torch.float)] +
            [torch.ones([env.state_dim]) * config.ignore_value] * (model.n_action_to_predict - 1)
        )

        # Get action from policy network
        with torch.no_grad():
            outputs = model.forward(images=image, states=states, cache=cache)

        cache = outputs.cache
        cache['cross_pos_ids'] += 1

        # Revert actions to numpy array and denormalize
        actions = (outputs.action_preds.float()[0, 0].cpu().numpy() * action_std + action_avg)
        if config.motion_option == 'global':
            vs, omegas = actions[:, :3], actions[:, 3:]
            for i in range(len(actions)):
                _, _, vs[i], omegas[i] = convert_to_local_frame(t_ref, q_ref, None, None, vs[i], omegas[i])
            actions = np.concatenate([vs, omegas], axis=1)
        # stop = outputs.stop_preds[0].item() > 0

        # Execute action in the environment
        observation, reward, terminated, truncated, info = env.step(actions)  # stop=stop)
        done = terminated or truncated

        # Update total reward and observation
        total_reward += reward
        crash = info['crash']
        seq_len = info['seq_len']

    # convert to video
    env.final_render(f'{run_name}_return{total_reward:.2f}_crash{crash}', mode='online_plus', re_render=config.re_render)

    return total_reward, crash, seq_len


def blender_simulation(model, logdir, config: BlenderSimulationConfig):
    # generated scenes
    infinigen_fpaths = {}
    for scene in sorted(os.listdir(infinigen_root)):
        if os.path.isdir(f'{infinigen_root}/{scene}'):
            for random_seed in sorted(os.listdir(f'{infinigen_root}/{scene}')):
                if os.path.isdir(f'{infinigen_root}/{scene}/{random_seed}'):
                    if os.path.exists(f'{infinigen_root}/{scene}/{random_seed}/frames/Image'):
                        if random_seed in ['0', '2b7ab387', '7c17e172', '5658d944']:
                            # skip the extra expensive ones
                            continue
                        if scene not in infinigen_fpaths:
                            infinigen_fpaths[scene] = []
                        infinigen_fpaths[scene].append(f'{infinigen_root}/{scene}/{random_seed}/fine/scene.blend')

    # google map scenes
    blosm_fpaths = {}
    for city in sorted(os.listdir(blosm_root)):
        if os.path.isdir(f'{blosm_root}/{city}'):
            blosm_fpaths[city] = [f'{blosm_root}/{city}/scene.blend']

    # all runs
    scene_fpaths = []
    for scene in blosm_fpaths:
        scene_fpaths.extend(blosm_fpaths[scene])
    for i in range(int(np.ceil((config.num_runs - len(blosm_fpaths)) / len(infinigen_fpaths)))):
        for scene in infinigen_fpaths:
            if i < len(infinigen_fpaths[scene]):
                scene_fpaths.append(infinigen_fpaths[scene][i])
    
    num_repeats = np.ones(len(scene_fpaths), dtype=int) * config.num_repeats
    
    results = []
    for i in tqdm.tqdm(range(min(config.num_runs, len(scene_fpaths)))):
        scene_fpath = scene_fpaths[i]
        if scene_fpath.startswith(blosm_root):
            run_name = scene_fpath.replace(blosm_root, '').split('/')[1]
        elif scene_fpath.startswith(infinigen_root):
            run_name = '_'.join(scene_fpath.replace(infinigen_root, '').split('/')[1:3])
        else:
            raise ValueError(f'Invalid scene fpath: {scene_fpath}.')

        with BlenderCameraEnv(
                scene_fpath=scene_fpath,
                fps=config.fps,
                action_fps=config.action_fps,
                run_dir=f'{logdir}/videos',
                resolution=config.resolution,
                video_duration=config.video_duration,
                motion_option=config.motion_option,
                cropped_sensor_width=config.cropped_sensor_width) as env:
            for j in range(num_repeats[i]):
                seed = i * 100 + j + 1
                total_reward, crash, seq_len = expand_episode(
                    env=env,
                    model=model,
                    run_name=run_name,
                    seed=seed,
                    random_init_pose=(j > 0),
                    config=config)
                results.append({
                    'render_fpath': scene_fpath,
                    'seed': seed,
                    'total_reward': total_reward,
                    'crash': crash,
                    'seq_len': seq_len
                })

    crash_rate = np.mean([result["crash"] is not None for result in results])
    avg_duration = np.mean([result["seq_len"] for result in results])
    print(f'Average return: {np.mean([result["total_reward"] for result in results])}\n'
          f'Crash rate: {crash_rate}\n'
          f'Sequence length: {avg_duration}\n')

    # save the crash rate as file
    with open(f'{logdir}/crash_{crash_rate}', 'w') as f:
        f.write(f'{crash_rate}')
    # save the average duration as file
    with open(f'{logdir}/duration_{avg_duration:.2f}', 'w') as f:
        f.write(f'{avg_duration}')

    return results


if __name__ == '__main__':
    from train import get_args_dict
    from configs.config_dvgmamba import DVGMambaConfig

    args_dict = get_args_dict()

    dvgmamba_config = DVGMambaConfig(args_dict)
    dvg_model = DVGMambaModel(dvgmamba_config).cuda().bfloat16()

    blender_config = BlenderSimulationConfig(logdir='logdir', run_name='run_name', args_dict=args_dict)
    blender_simulation(dvg_model, '/home/jinpeng-yu/Desktop/dvgmamba/logs/unit_test', blender_config)
