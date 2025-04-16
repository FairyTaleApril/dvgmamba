import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import os
import time

import torch
import torch.profiler
from tqdm import tqdm
from transformers import set_seed
from mamba_ssm.utils.generation import InferenceParams

from configs.config_drone_path_dataset import DronePathDatasetConfig
from configs.config_dvgmamba import DVGMambaConfig
from data.drone_path_dataset import DronePathDataset
from speed_models.modeling_dvgmamba import DVGMambaModel


def main():
    run_name = f'Mamba {datetime.datetime.now().strftime("%m-%d %H-%M")}'
    logdir = f'logs/{run_name}'
    os.makedirs(logdir, exist_ok=True)

    args_dict = get_args_dict()
    set_seed(args_dict['seed'])
    num_runs = args_dict['num_runs']
    device =args_dict['device']

    drone_path_dataset_config = DronePathDatasetConfig(args_dict=args_dict)
    dataset = DronePathDataset(config=drone_path_dataset_config)

    dvgmamba_config = DVGMambaConfig(args_dict=args_dict)
    model = DVGMambaModel(config=dvgmamba_config).to(dvgmamba_config.dtype).to(device)
    model.eval()
    model.print_num_parameters()

    batch_size, seq_length = 1, 1
    images = torch.randn([batch_size, seq_length, 3, dataset.resolution[0], dataset.resolution[1]], device=device)
    states = torch.randn([seq_length, dvgmamba_config.state_dim], device=device)
    cache = {
        'cross_pos_ids': 0,
        'all_input_embeddings': None,
        'inference_params': None
        # 'inference_params': InferenceParams(max_seqlen=dvgmamba_config.max_model_frames, max_batch_size=1)
    }

    # t0 = time.time()
    # for repeat in tqdm(range(num_runs)):
    #     with torch.no_grad():
    #         model.forward(images=images, states=states, cache=cache, test_speed=True)
    # 
    # t1 = time.time()
    # print(f'inference num: {num_runs}, time: {t1 - t0}')
    # print(f'inference speed: {num_runs / (t1 - t0)}')
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            model(images=images, states=states, cache=cache, test_speed=True)
            
    print(prof.key_averages().table(sort_by="flops", row_limit=10))
    total_flops = sum([item.flops for item in prof.key_averages()])
    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")


def get_args_dict():
    parser = argparse.ArgumentParser(description='Training script')

    # Global settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='logs/Mamba 03-18 06-59')
    # 'OPTIX': on RTX GPUs, faster but more likely to cause glitches
    # 'OPENIMAGEDENOISE': on CPUs, slower but more stable
    parser.add_argument('--default_denoiser', type=str, default='OPENIMAGEDENOISE', choices=['OPTIX', 'OPENIMAGEDENOISE'])

    # Dataset settings
    parser.add_argument('--root', type=str, default='/workspace/DVG')
    # parser.add_argument('--hdf5_fname', type=str, default='dataset_mini.h5')
    # parser.add_argument('--hdf5_fname', type=str, default='dataset_2k.h5')
    parser.add_argument('--hdf5_fname', type=str, default='dataset_mini.h5')

    # Model settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='torch.bfloat16')
    parser.add_argument('--n_layer', type=int, default=22)
    parser.add_argument('--use_depth', type=bool, default=True)
    parser.add_argument('--max_model_frames', type=int, default=150)
    parser.add_argument('--fix_image_width', type=bool, default=True)
    parser.add_argument('--prediction_option', type=str, default='iterative', choices=['iterative', 'one-shot'])

    parser.add_argument('--num_runs', type=int, default=1000)

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == '__main__':
    main()
