# encoding: utf-8
"""
# Last Change:  2022-07-12 10:52:59
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from PIL import Image
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import read_image
from torchvision.transforms import functional as F
from .transforms.transforms import ToTensor

class _CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path, format="RGB")  # 0-255, PIL Image

        if self.transform is not None:
            img = self.transform(img)  # 0-255, tensor

        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]

        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "flipped": False,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)

def block_erasing(img, sl=0.125, sh=0.25, mean=(123.675, 116.28, 103.53)):
    img = np.asarray(img, dtype=np.float32).copy()

    img_h = img.shape[0]
    h = int(round(random.uniform(sl, sh) * img_h))
    x1 = random.randint(0, img.shape[0] - h)

    if img.shape[2] == 3:
        img[x1:x1 + h, :, 0] = mean[0]
        img[x1:x1 + h, :, 1] = mean[1]
        img[x1:x1 + h, :, 2] = mean[2]
    else:
        img[x1:x1 + h, :, 0] = mean[0]
    return img, torch.tensor([x1, x1 + h])

class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

        self.to_tensor = ToTensor()
        self.training = False
        self.do_rse = False  # TODO-ADD 20220712
        self.do_flip = False

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path, format="RGB")  # 0-255, PIL Image

        flipped = False
        erased = torch.tensor([-1, -1])
        if self.transform is not None:
            if random.random() < 0.5 and self.training and self.do_flip:
                flipped = True
                img = F.hflip(img)

            img = self.transform(img)  # 0-255, tensor

            if random.random() < 0.5 and self.training and self.do_rse:
                img, erased = block_erasing(img)

            img = self.to_tensor(img)  # 0-255, tensor

        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]

        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "flipped": flipped,
            "erased": erased,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
