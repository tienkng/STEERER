import os
import pprint
import sys
import _init_paths
from lib.core.Counter import Counter
from lib.utils.utils import create_logger, random_seed_setting
from lib.utils.modelsummary import get_model_summary
from lib.core.cc_function import test_loc
import datasets
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import timeit
import logging
import argparse
from lib.models.build_counter import Baseline_Counter
from lib.utils.dist_utils import (
    get_dist_info,
    init_dist)
from mmcv import Config, DictAction
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Test crowd counting network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--checkpoint', help='experiment configure file name', required=True, type=str)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi', 'torchrun'], default='none', help='job launcher')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config...')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def read_box_gt(box_gt_file):
    gt_data = {}
    with open(box_gt_file) as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            line_data = [int(i) for i in line]
            idx, num = [line_data[0], line_data[1]]
            points_r = []
            if num > 0:
                points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
                gt_data[idx] = {'num': num, 'points': points_r[:, 0:2], 'sigma': points_r[:, 2:4], 'level': points_r[:, 4]}
            else:
                gt_data[idx] = {'num': 0, 'points': [], 'sigma': [], 'level': []}
    return gt_data

def main():
    args = parse_args()
    config = Config.fromfile(args.cfg)
    if args.cfg_options is not None:
        config.merge_from_dict(args.cfg_options)

    logger, final_output_dir = create_logger(config, args.cfg, 'test')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    logger.info(f'Dataset root: {config.dataset.root}')

    logger.info('GPU idx: ' + os.environ.get('CUDA_VISIBLE_DEVICES', 'None'))
    gpus = config.gpus
    distributed = torch.cuda.device_count() > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        init_dist(args.launcher)
        if args.launcher == 'pytorch':
            args.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            rank, world_size = get_dist_info()
            args.local_rank = rank
    
    random_seed_setting(config)
    model = Baseline_Counter(config.network, config.dataset.den_factor, config.train.route_size, device)
    logger.info(f'Model created: {model.__class__.__name__}')

    if args.checkpoint:
        model_state_file = args.checkpoint
    else:
        model_state_file = config.test.model_file or os.path.join(final_output_dir, 'final_state.pth')
    logger.info(f'=> loading model from {model_state_file}')

    pretrained_dict = torch.load(model_state_file, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict, strict=False)
    model = model.to(device)

    test_dataset = eval('datasets.' + config.dataset.name)(
        root=config.dataset.root,
        list_path=None,
        num_samples=None,
        multi_scale=False,
        flip=False,
        base_size=config.test.loc_base_size,
        downsample_rate=1)
    logger.info(f'Dataset created with {len(test_dataset)} samples')

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True)

    start = timeit.default_timer()
    
    outputs = test_loc(config, test_dataset, testloader, model,
                       test_dataset.mean, test_dataset.std,
                       sv_dir=final_output_dir, sv_pred=True)
    

    end = timeit.default_timer()
    logger.info(f'Total testing time: {end - start:.2f} seconds')

if __name__ == '__main__':
    main()
    
    

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
