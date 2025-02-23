import argparse
import datetime
import os
import warnings

import torch
from transformers import Trainer, TrainingArguments, set_seed

import wandb
from configs.config_blender_simulation import BlenderSimulationConfig
from configs.config_drone_path_dataset import DronePathDatasetConfig
from configs.config_dvgmamba import DVGMambaConfig
from data.drone_path_dataset import DronePathDataset, collate_fn_video_drone_path_dataset
from blender_simulation_mamba import blender_simulation
from models.modeling_dvgmamba import DVGMambaModel

warnings.filterwarnings("ignore")


def main():
    args_dict = get_args_dict()

    set_seed(args_dict.get('seed', 42))

    run_name = f'Mamba {datetime.datetime.now().strftime("%m-%d %H-%M")}'
    logdir = f'logs/{run_name}'
    os.makedirs(logdir, exist_ok=True)
    print(logdir)

    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project='dvgmamba', name=run_name)

    if args_dict['load_checkpoint']:
        checkpoint_path = args_dict['checkpoint_path']
        checkpoint_dirs = [f for f in os.listdir(checkpoint_path) if
                           'checkpoint' in f and os.path.isdir(os.path.join(checkpoint_path, f))]
        checkpoint_fpath = f"{args_dict['checkpoint_path']}/{checkpoint_dirs[0]}/pytorch_model.bin"

        dvgmamba_config = DVGMambaConfig(args_dict=args_dict)
        dvgmamba_config.load_config(fpath=f'{checkpoint_path}/dvgmamba_config.json')

        blender_simulation_config = BlenderSimulationConfig(logdir=logdir, run_name=run_name, args_dict=args_dict)
        blender_simulation_config.load_config(fpath=f'{checkpoint_path}/blender_simulation_config.json')
        blender_simulation_config.logdir = logdir
        blender_simulation_config.run_name = run_name

        model = DVGMambaModel(config=dvgmamba_config).to(dvgmamba_config.dtype)
        model.print_num_parameters()
        model.load_model(checkpoint_fpath)
    else:
        dvgmamba_config = DVGMambaConfig(args_dict=args_dict)
        dvgmamba_config.save_config(fpath=f'{logdir}/dvgmamba_config.json')

        drone_path_dataset_config = DronePathDatasetConfig(args_dict=args_dict)
        drone_path_dataset_config.save_config(fpath=f'{logdir}/drone_path_dataset_config.json')

        blender_simulation_config = BlenderSimulationConfig(logdir=logdir, run_name=run_name, args_dict=args_dict)
        blender_simulation_config.save_config(fpath=f'{logdir}/blender_simulation_config.json')

        model = DVGMambaModel(config=dvgmamba_config).to(dvgmamba_config.dtype)
        model.print_num_parameters()

        train_dataset = DronePathDataset(config=drone_path_dataset_config)

        training_args = TrainingArguments(
            output_dir=logdir,
            run_name=run_name,

            num_train_epochs=args_dict['epochs'],
            per_device_train_batch_size=args_dict['batch_size'],
            gradient_accumulation_steps=args_dict['gradient_accumulation_steps'],

            learning_rate=args_dict['learning_rate'],
            lr_scheduler_type='cosine',
            warmup_ratio=0.03,

            dataloader_num_workers=args_dict['num_workers'],
            dataloader_drop_last=True,
            logging_steps=args_dict['logging_steps'],

            save_safetensors=False,
            bf16=True,
            tf32=True,

            report_to='all',
            save_strategy="epoch",
            save_total_limit=1,
        )
        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn_video_drone_path_dataset
        )

        trainer.train()
        trainer.save_model()
        del trainer

        # clean up cuda memory
        torch.cuda.empty_cache()

    blender_simulation(model=model, logdir=logdir, config=blender_simulation_config)


def get_args_dict():
    parser = argparse.ArgumentParser(description='Training script')

    # Global settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='logs/Mamba 02-23 22-32')

    # BaseConfig settings
    parser.add_argument('--motion_option', type=str, default='local', choices=['local', 'global'])

    # Dataset settings
    parser.add_argument('--root', type=str, default='/media/jinpeng-yu/Data1/DVG')
    parser.add_argument('--hdf5_fname', type=str, default='dataset_mini.h5')
    # parser.add_argument('--hdf5_fname', type=str, default='dataset_2k_fpv.h5')
    # augmentation settings
    parser.add_argument('--random_horizontal_flip', type=bool, default=False)
    parser.add_argument('--random_scaling', type=bool, default=False)
    parser.add_argument('--random_temporal_crop', type=bool, default=False)
    parser.add_argument('--random_color_jitter', type=bool, default=False)

    # Model settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='torch.bfloat16')
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--use_depth', type=bool, default=True)
    parser.add_argument('--max_model_frames', type=int, default=150)
    parser.add_argument('--fix_image_width', type=bool, default=True)
    parser.add_argument('--prediction_option', type=str, default='iterative', choices=['iterative', 'one-shot'])
    # Token settings
    parser.add_argument('--n_token_state', type=int, default=1)
    parser.add_argument('--n_token_boa', type=int, default=1)
    parser.add_argument('--n_token_action', type=int, default=1)
    # loss coef settings
    parser.add_argument('--loss_coef_state', type=float, default=0)
    parser.add_argument('--loss_coef_action', type=float, default=1)
    parser.add_argument('--loss_coef_stop', type=float, default=0)

    # Training settings
    parser.add_argument('--epochs', type=int, default=2)  # 5
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--logging_steps', type=int, default=50)  # 50

    # Simulation settings
    parser.add_argument('--num_runs', type=int, default=2)
    parser.add_argument('--num_repeats', type=int, default=3)
    parser.add_argument('--re_render', type=bool, default=True)

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == '__main__':
    main()
