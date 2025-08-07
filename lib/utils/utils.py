# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import shutil

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)


    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val


def random_seed_setting(config):

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    seed = config.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPUS).strip('(').strip(')')


def copy_cur_env(work_dir, dst_dir, exception):
    # if os.path.exists(dst_dir):
    #     shutil.rmtree(dst_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)
def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.log_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.dataset.name
    model = cfg.network.backbone+'_'+cfg.network.sub_arch
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)

    if phase=='test':
        test_output_dir = root_output_dir / dataset / phase /cfg_name
        print('=> creating {}'.format(test_output_dir))
        test_output_dir.mkdir(parents=True, exist_ok=True)


        final_log_file = test_output_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger, str(test_output_dir)

    elif phase == 'train':
        resume_path = cfg.train.resume_path

        val_output_dir =  Path(cfg.log_dir) / dataset / model / 'val'
        val_output_dir.mkdir(parents=True, exist_ok=True)

        if resume_path is not None:
            train_log_dir = resume_path
            final_log_file = Path(train_log_dir) / (os.path.basename(train_log_dir)+'_train.log')
        else:
            train_log_dir = Path(cfg.log_dir) / dataset / model / \
                    (cfg_name + '_' + time_str)
            print('=> creating {}'.format(train_log_dir))
            train_log_dir.mkdir(parents=True, exist_ok=True)

            final_log_file = Path(train_log_dir) / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger,  str(train_log_dir)
    else:
        raise ValueError('phase must be "test" or "train"')


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

def save_results_more(name, sv_dir, input_img, pre_map, gt_map, pre_cnt, gt_cnt, pre_points=None, gt_points=None):
    logger = logging.getLogger(__name__)
    name = name[0] if isinstance(name, (list, tuple)) else name  # Lấy tên ảnh gốc
    os.makedirs(sv_dir, exist_ok=True)  # Tạo thư mục sv_dir nếu chưa tồn tại

    # Chuyển input_img (NumPy array [H, W, C], BGR) thành ảnh PIL (RGB)
    if isinstance(input_img, np.ndarray):
        if input_img.shape[2] in [1, 3]:  # [H, W, C]
            # Chuẩn hóa giá trị pixel: nhân với 255 nếu giá trị trong [0, 1]
            input_img = (input_img * 255).astype(np.uint8) if input_img.max() <= 1.0 else input_img.astype(np.uint8)
            # Chuyển từ BGR sang RGB
            if input_img.shape[2] == 3:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            pil_input = Image.fromarray(input_img)
        else:
            logger.error(f"Invalid input_img shape: {input_img.shape}")
            return
    else:
        logger.error(f"Expected input_img to be np.ndarray, got {type(input_img)}")
        return

    # Vẽ điểm dự đoán lên ảnh gốc
    img_with_points = np.array(pil_input)  # [H, W, C], RGB
    RGB_G = (0, 255, 0)  # Xanh lá cho điểm dự đoán (RGB)
    thickness = 5
    marker_size = 20

    if pre_points is not None and len(pre_points) > 0:
        for point in pre_points:
            point = point.astype(np.int32)
            x, y = point[0], point[1]
            cv2.drawMarker(img_with_points, (x, y), RGB_G, markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=thickness)

    # Chuyển lại thành PIL để lưu
    pil_img_with_points = Image.fromarray(img_with_points)  # Đã là RGB

    # Lưu ảnh với tên giống ảnh gốc
    output_path = os.path.join(sv_dir, f'{name}.jpg')
    pil_img_with_points.save(output_path)
    logger.info(f'Image with points saved to {output_path}')