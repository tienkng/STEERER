
import os

import cv2
import numpy as np
from PIL import Image
import json
import torch
from torch.nn import functional as F
import random
from .base_dataset import BaseDataset
from .nwpu import NWPU
import glob

class SHHB(NWPU):
    def __init__(self,
                 root,
                 list_path=None,
                 num_samples=None,
                 num_classes=1,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 min_unit = (32,32),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=(0.5, 1/0.5),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(SHHB, self).__init__(
            root,
            list_path,
            num_samples,
            num_classes,
            multi_scale,
            flip,
            ignore_label,
            base_size,
            crop_size,
            min_unit ,
            center_crop_test,
            downsample_rate,
            scale_factor,
            mean,
            std)
        
    def read_files(self):
        files = []
        if self.list_path is None:
            # Quét trực tiếp thư mục root
            image_extensions = ['*.jpg', '*.jpeg', '*.png']
            image_list = []
            for ext in image_extensions:
                image_list.extend(glob.glob(os.path.join(self.root, ext)))
            
            if not image_list:
                raise ValueError(f"No images found in {self.root}")
            
            image_list.sort()
            for img_path in image_list:
                image_id = os.path.basename(img_path)
                files.append({
                    "img": image_id,
                    "label": None,
                    "name": os.path.splitext(image_id)[0],
                    "weight": 1
                })
        else:
            # Logic cũ: đọc từ list_path
            val_gt_path = os.path.join(self.root, 'val_gt_loc.txt')
            if os.path.exists(val_gt_path):
                box_gt_Info = self.read_box_gt(val_gt_path)
            else:
                box_gt_Info = []
            
            for item in self.img_list:
                image_id = item[0]
                if 'val' in self.list_path:
                    self.box_gt.append(box_gt_Info[int(image_id)])
                files.append({
                    "img": 'images/' + image_id + '.jpg',
                    "label": 'jsons/' + image_id + '.json',
                    "name": image_id,
                    "weight": 1
                })
        return files