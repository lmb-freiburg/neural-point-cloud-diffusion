#!/usr/bin/env python3

import argparse
import sys
import os
import os.path as osp

import torch

from npcd.data import create_dataset
from npcd.models import NPCD
from npcd.eval import PointNeRFEvaluation
from npcd.utils import set_seed, writer, logging, load_config, print_config, cp


@torch.no_grad()
def eval(args):
    config = load_config(args.config)
    
    out_dir = args.output
    if out_dir is not None:
        log_file_path = osp.join(out_dir, "log.txt")
        os.makedirs(out_dir, exist_ok=True)
        logging.add_log_file(log_file_path, flush_line=True)

        with open(osp.join(out_dir, "cmd.txt"), 'a') as f:
            f.write("python " + " ".join(sys.argv) + "\n")
        cp(args.config, osp.join(out_dir, "config.yaml"), verbose=False)
    
    log_dir = args.log_dir if args.log_dir is not None else out_dir
    if log_dir is not None:
        tensorboard_log_dir = osp.join(log_dir, "tensorboard_logs")
        wandb_log_dir = osp.join(log_dir, "wandb_logs")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        os.makedirs(wandb_log_dir, exist_ok=True)
        writer.setup_writers(log_tensorboard=not args.no_tensorboard, 
                            log_wandb=args.wandb, 
                            tensorboard_logs_dir=tensorboard_log_dir, 
                            wandb_logs_dir=wandb_log_dir,
                            exp_id=args.exp_id,
                            comment=args.comment,)
        
    set_seed(args.seed)

    logging.info()
    logging.info(f"Evaluating the PointNeRF autodecoder with:\n\tconfig: {args.config}\n\tweights: {args.weights}\n\toutput directory: {args.output}\n")
    
    print("Config:")
    print_config(config)
    print()
    
    dataset = create_dataset(name=config["train_dataset"])
    
    model = NPCD(pointnerf_only=True, **config["model"])
    model.load_state_dict(torch.load(args.weights), strict=False)
    model.eval().cuda()

    eval = PointNeRFEvaluation(out_dir=out_dir, eval_batch_size=args.eval_batch_size)
    
    samples = args.num_samples if args.num_samples is not None else args.samples
    qualitatives = args.qualitatives if args.qualitatives is not None else args.num_qualitatives

    eval(dataset=dataset, model=model, samples=samples, qualitatives=qualitatives,
         eval_name=args.eval_name, finished_iterations=args.finished_iterations,)
    
    if out_dir is not None:
        logging.remove_log_file(log_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help="Path to folder for output data.")
    parser.add_argument('--config', help="Path to config file.", required=True)
    parser.add_argument('--weights', help="Path to weights of the model.", required=True)

    parser.add_argument('--seed', type=int, help="Random seed. Default: 42.", default=42)
    parser.add_argument('--eval_batch_size', type=int, help="Batch size for evaluation. Runtime measurements are "
                        "only available when using batch size 1. Default: 8.", default=8)
    
    parser.add_argument('--eval_name', help="Name of the evaluation. Optional.", type=str)
    parser.add_argument('--finished_iterations', type=int, help="Number of iterations that the model was trained. Only used for logging. Optional.")

    parser.add_argument('--num_samples', type=int, help='Number of samples to be evaluated. Default: evaluate all.')
    parser.add_argument('--samples', type=int, nargs='*',
                        help='Index of sample that should be evaluated. Ignored if num_samples is used. '
                             'Default: evaluate all.')

    parser.add_argument('--num_qualitatives', type=int, default=10,
                        help='Number of qualitatives to be output. Negative values output all qualitatives. '
                             'Ignored if --qualitative is used. Default: 10.')
    parser.add_argument('--qualitatives', type=int, nargs='*',
                        help='Index of sample where qualitatives should be output.')

    parser.add_argument('--log_dir', help="Path to folder for tensorboard and wandb logs. Optional. Default: use output dir.")
    parser.add_argument('--no_tensorboard', action='store_true', help='Do not log to tensorboard. Default: do log.')
    parser.add_argument('--wandb', action='store_true', help='Log to weights and biases. Default: Do not log.')
    parser.add_argument('--exp_id', type=str, help="Experiment ID. Used for wandb logging.")
    parser.add_argument('--comment', type=str, help="Comment for the experiment. Used for wandb logging.")

    args = parser.parse_args()

    eval(args)
