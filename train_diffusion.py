#!/usr/bin/env python3

import argparse
import sys
import os
import os.path as osp

import torch

from npcd.utils import set_seed, writer, logging, load_config, print_config, cp
from npcd.models import NPCD
from npcd.train import DiffusionTraining
from npcd.data import PointNeRFDataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(args):
    out_dir = args.output
    
    config = load_config(args.config)
    
    tensorboard_logs_dir = osp.join(out_dir, "tensorboard_logs")
    wandb_logs_dir = osp.join(out_dir, "wandb_logs")
    log_file_path = osp.join(out_dir, "log.txt")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tensorboard_logs_dir, exist_ok=True)
    os.makedirs(wandb_logs_dir, exist_ok=True)
    writer.setup_writers(log_tensorboard=not args.no_tensorboard, 
                         log_wandb=args.wandb, 
                         tensorboard_logs_dir=tensorboard_logs_dir, 
                         wandb_logs_dir=wandb_logs_dir,
                         exp_id=args.exp_id,
                         comment=args.comment,)
    logging.add_log_file(log_file_path, flush_line=True)

    with open(osp.join(args.output, "cmd.txt"), 'a') as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    cp(args.config, osp.join(out_dir, "config.yaml"), verbose=False)
    
    set_seed(args.seed)
        
    logging.info()
    logging.info(f"Training the NPCD diffusion model with:\n\tconfig: {args.config}\n\tpointnerf weights: {args.pointnerf_weights}\n\toutput directory: {args.output}.\n")
        
    print("Config:")
    print_config(config)
    print()
    
    model = NPCD(freeze_pointnerf=True, use_flash_attn=(args.dtype == "float16" or args.dtype == "bfloat16"), **config["model"])
    model.load_state_dict(torch.load(args.pointnerf_weights), strict=False)
    model.train().cuda()
    
    dataset = PointNeRFDataset(pointnerf=model.pointnerf)
    
    training = DiffusionTraining(out_dir=out_dir,
                                 model=model,
                                 dataset=dataset,
                                 dtype=args.dtype,
                                 num_workers=args.num_workers,
                                 verbose=True,
                                 **config["diffusion_training"])

    training()
    
    logging.remove_log_file(log_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help="Path to folder for output data.", required=True)
    parser.add_argument('--config', help="Path to config file.", required=True)
    parser.add_argument('--pointnerf_weights', help="Path to config file.", required=True)
    
    parser.add_argument('--dtype', type=str, help="Data type for training. Options: float32, float16, bfloat16. Default: float16.", default="float16")
    
    parser.add_argument('--seed', type=int, help="Random seed. Default: 42.", default=42)
    parser.add_argument('--num_workers', type=int, help="Number of workers for data loading. Default: 8.", default=8)

    parser.add_argument('--no_tensorboard', action='store_true', help='Do not log to tensorboard. Default: do log.')
    parser.add_argument('--wandb', action='store_true', help='Log to weights and biases. Default: Do not log.')
    parser.add_argument('--exp_id', type=str, help="Experiment ID. Used for wandb logging.")
    parser.add_argument('--comment', type=str, help="Comment for the experiment. Used for wandb logging.")

    args = parser.parse_args()

    train(args)
