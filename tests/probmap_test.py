#!/usr/bin/env python
# encoding: utf-8
# Last Change:  2022-08-28 18:13:41

import sys
import time
import random
import subprocess

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

import os
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torchvision import utils
import torch.nn.functional as F

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def class_activation_map_on_im(org_im, activation_map, out_size=(384, 192)):
    '''
    Args:
        org_im: cv2 image, BGR, 0-255
        activation_map: target class activation map (grayscale) 0-255
        out_size: desired output image size (height width)

    Returns:
        numpy array of shape HWC, 0-255, uint8
    '''
    org_im = cv2.resize(org_im, (out_size[1], out_size[0]))
    activation_map = cv2.resize(activation_map, (out_size[1], out_size[0]))

    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    # Heatmap on picture
    im_with_heatmap = np.float32(activation_heatmap) + np.float32(org_im)
    max_val = np.clip(np.max(im_with_heatmap), a_min=1e-10, a_max=None)
    im_with_heatmap = im_with_heatmap / max_val
    im_with_heatmap = im_with_heatmap[..., ::-1]  # BGR to RGB
    return np.uint8(255 * im_with_heatmap)

def cv2im_transformer(
        cv2im,
        im_size=(256, 128),
        requires_grad=False,
        norm=True,
        div=True):
    """ Resize, to_tensor and ImageNet normalized

    Args:
        cv2im: Image to process, 0-255, BGR, (H,W,C)
        im_size: (height, width) image size after resizing
    returns:
        im_as_ten(tensor): float tensor of shape 1CHW, 0-1
    """
    # mean and std for channels (ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # resize image
    cv2im = cv2.resize(cv2im, (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR)  # bilinear
    im_as_arr = np.array(cv2im)  # HWC
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])  # BGR to RGB
    # normalizing 0-1
    if div:
        im_as_arr = (im_as_arr / 255)  # HWC

    # normalize
    if norm:
        mean = np.array(mean)
        std = np.array(std)
        im_as_arr = (im_as_arr - mean) / std  # HWC

    im_as_ten = torch.tensor(im_as_arr, dtype=torch.float, requires_grad=requires_grad).contiguous()
    im_as_ten = im_as_ten.permute(2, 0, 1).contiguous()  # HWC to CHW
    im_as_ten.unsqueeze_(0)  # 1CHW
    return im_as_ten

def plot_prob_maps_with_label(model, image_items, in_size, save_dir=None):
    model.eval()
    MAX_NUM_ID = -1
    MAX_IMG_PER_ID = 1

    # collect image paths of all pids
    pid_to_fpath = defaultdict(list)
    for i, item in enumerate(image_items):
        # item(tuple): (img_path, pid, camid), pid = dataset_name + "_" + str(pid)
        pid = item[1]
        pid_to_fpath[pid].append(item[0])

    # Go through person identities
    for ii, (pid, fpath) in enumerate(pid_to_fpath.items()):
        if MAX_NUM_ID > 0 and ii >= MAX_NUM_ID:
            break

        print('Processing images of {} ({}/{}) ...'.format(pid, ii + 1, len(pid_to_fpath)))
        # list of cv2 images of a pid
        original_ims = [cv2.imread(x, cv2.IMREAD_COLOR) for x in fpath]
        # resize, to_tensor, normalize cv2im and concatenate as an input batch
        batch_images = torch.cat(
            [cv2im_transformer(x, im_size=in_size, norm=False, div=False) for x in original_ims],
            dim=0
        )
        batch_images = batch_images.cuda()  # BCHW
        raw_batch_images = batch_images.clone()

        #
        if batch_images.size(0) >= MAX_IMG_PER_ID:
            batch_images = batch_images[:MAX_IMG_PER_ID, ...]
            raw_batch_images = raw_batch_images[:MAX_IMG_PER_ID, ...]
            fpath = fpath[:MAX_IMG_PER_ID]

        # forward at eval mode
        model_input = {
            "images": batch_images,
            "img_paths": fpath,
            "flipped": [False] * len(fpath),
            "erased": [torch.tensor([-1, -1])] * len(fpath)
        }
        output = model(model_input)

        # (b num_parts h w) to (b num_parts H W)
        # import pdb; pdb.set_trace()
        prob_maps = F.interpolate(output['prob_maps'], size=batch_images.size()[-2:],
                                  mode='bilinear', align_corners=False)
        part_prob_maps = prob_maps[:, 2:, ...]
        part_prob_maps = part_prob_maps / part_prob_maps.max(dim=1, keepdim=True)[0] * 255.
        fg_prob_map = prob_maps[:, 1:2, ...]
        fg_prob_map = fg_prob_map / fg_prob_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] * 255.
        fg_mask_gt = prob_maps[:, 0:1, ...]
        fg_mask_gt = fg_mask_gt * 255.
        prob_maps = torch.cat([fg_mask_gt, fg_prob_map, part_prob_maps[:, 0:, ...]], dim=1)
        assert raw_batch_images.size(0) == prob_maps.size(0)

        result_images = []
        for ii in range(raw_batch_images.size(0)):
            single_image = [raw_batch_images[ii]]  # 3HW
            part_probs = prob_maps[ii].split(dim=0, split_size=1)  # List[Tensor(1HW)]
            part_probs = [item.expand_as(batch_images[ii]) for item in part_probs]  # List[Tensor(3HW)]
            single_image += part_probs
            result_images += single_image

        grid = utils.make_grid(result_images, nrow=prob_maps.size(1) + 1, padding=2, normalize=False,
                               range=None, scale_each=False, pad_value=255)
        ndarr = grid.mul(1).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        filename = os.path.join(save_dir, 'prob_{}.jpg'.format(pid))
        im.save(filename)
    return

def plot_prob_maps(model, image_items, in_size, save_dir=None):
    model.eval()
    MAX_NUM_ID = -1
    MAX_IMG_PER_ID = 64

    # collect image paths of all pids
    pid_to_fpath = defaultdict(list)
    for i, item in enumerate(image_items):
        # item(tuple): (img_path, pid, camid), pid = dataset_name + "_" + str(pid)
        pid = item[1]
        #if not("dukemtmc_54"==pid or "dukemtmc_202"==pid):
        #    continue
        pid_to_fpath[pid].append(item[0])
    #random.shuffle(pid_to_fpath)
    # Go through person identities
    for ii, (pid, fpath) in enumerate(pid_to_fpath.items()):
        if MAX_NUM_ID > 0 and ii >= MAX_NUM_ID:
            break

        print('Processing images of {} ({}/{}) ...'.format(pid, ii + 1, len(pid_to_fpath)))
        if MAX_IMG_PER_ID > 0:
            #fpath = fpath if len(fpath)<=MAX_IMG_PER_ID else random.sample(fpath, MAX_IMG_PER_ID)
            fpath = fpath if len(fpath)<=MAX_IMG_PER_ID else fpath[:MAX_IMG_PER_ID]

        # list of cv2 images of a pid
        original_ims = [cv2.imread(x, cv2.IMREAD_COLOR) for x in fpath]
        # resize, to_tensor, normalize cv2im and concatenate as an input batch
        batch_images = torch.cat(
            [cv2im_transformer(x, im_size=in_size, norm=False, div=False) for x in original_ims],
            dim=0
        )
        batch_images = batch_images.cuda()  # BCHW
        raw_batch_images = batch_images.clone()

        # forward at eval mode
        model_input = {
            "images": batch_images,
            "img_paths": fpath,
            "flipped": [False] * len(fpath),
            "erased": [torch.tensor([-1, -1])] * len(fpath)
        }
        output = model(model_input)

        # (b num_parts h w) to (b num_parts H W)
        prob_maps = F.interpolate(output['part_maps'], size=batch_images.size()[-2:],
                                  mode='bilinear', align_corners=False)
        part_prob_maps = prob_maps[:, 1:, ...]
        part_prob_maps = part_prob_maps[:, (0,1,3), ...]
        
        _b, _num_part, _h = part_prob_maps.size()[:3]
        tau = 5.
        part_prob_maps = F.softmax(part_prob_maps.view(_b,_num_part, -1)*tau,dim=-1).view(_b, _num_part, _h, -1)

        #p_mask = part_prob_maps.max(dim=2, keepdim=True)[0].max(dim=3,keepdim=True)[0]  # (b num_parts 1 w)
        #part_prob_maps = part_prob_maps/p_mask * 255.

        max_val = part_prob_maps.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        min_val = part_prob_maps.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        val_range = torch.clamp(max_val-min_val, min=1e-10)
        part_prob_maps = (part_prob_maps - min_val) / val_range * 255.

        prob_maps = part_prob_maps
        assert prob_maps.size(0) == len(original_ims)

        result_images = []
        for ii in range(prob_maps.size(0)):
            single_image = original_ims[ii]  # cv2img
            _h, _w = prob_maps.size(2), prob_maps.size(3)
            #part_probs = prob_maps[ii].split(dim=0, split_size=1)  # List[Tensor(1HW)]
            #part_probs = [item.expand(3,_h,_w) for item in part_probs]  # List[Tensor(3HW)]

            part_probs = [_t.detach().cpu().numpy().astype(np.uint8) for _t in prob_maps[ii]]
            _cur = [class_activation_map_on_im(single_image, _t) for _t in part_probs]
            _cur = [torch.tensor(_t,dtype=torch.float).permute(2,0,1).contiguous() for _t in _cur]
            result_images += _cur

        grid = utils.make_grid(result_images, nrow=prob_maps.size(1) + 0, padding=2, normalize=False,
                               range=None, scale_each=False, pad_value=255)
        ndarr = grid.mul(1).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        filename = os.path.join(save_dir, "hard_pr_query", '{}.jpg'.format(pid))
        im.save(filename)
    return

def plot_multilevel_prob_maps(model, image_items, in_size, save_dir=None):
    model.eval()
    MAX_NUM_ID = -1
    MAX_IMG_PER_ID = 64

    # collect image paths of all pids
    pid_to_fpath = defaultdict(list)
    for i, item in enumerate(image_items):
        # item(tuple): (img_path, pid, camid), pid = dataset_name + "_" + str(pid)
        pid = item[1]
        #if not("dukemtmc_54"==pid or "dukemtmc_202"==pid):
        #    continue
        pid_to_fpath[pid].append(item[0])
    #random.shuffle(pid_to_fpath)
    # Go through person identities
    for ii, (pid, fpath) in enumerate(pid_to_fpath.items()):
        if MAX_NUM_ID > 0 and ii >= MAX_NUM_ID:
            break

        print('Processing images of {} ({}/{}) ...'.format(pid, ii + 1, len(pid_to_fpath)))
        if MAX_IMG_PER_ID > 0:
            #fpath = fpath if len(fpath)<=MAX_IMG_PER_ID else random.sample(fpath, MAX_IMG_PER_ID)
            fpath = fpath if len(fpath)<=MAX_IMG_PER_ID else fpath[:MAX_IMG_PER_ID]

        # list of cv2 images of a pid
        original_ims = [cv2.imread(x, cv2.IMREAD_COLOR) for x in fpath]
        # resize, to_tensor, normalize cv2im and concatenate as an input batch
        batch_images = torch.cat(
            [cv2im_transformer(x, im_size=in_size, norm=False, div=False) for x in original_ims],
            dim=0
        )
        batch_images = batch_images.cuda()  # BCHW
        raw_batch_images = batch_images.clone()

        # forward at eval mode
        model_input = {
            "images": batch_images,
            "img_paths": fpath,
            "flipped": [False] * len(fpath),
            "erased": [torch.tensor([-1, -1])] * len(fpath)
        }
        output = model(model_input)

        # (b num_parts h w) to (b num_parts H W)
        prob_maps = F.interpolate(output['fg_map'], size=batch_images.size()[-2:],
                                  mode='bilinear', align_corners=False)
        part_prob_maps = prob_maps[:, (0,1,3), ...]
        
        _b, _num_part, _h = part_prob_maps.size()[:3]

        #p_mask = part_prob_maps.max(dim=2, keepdim=True)[0].max(dim=3,keepdim=True)[0]  # (b num_parts 1 w)
        #part_prob_maps = part_prob_maps/p_mask * 255.

        max_val = part_prob_maps.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        min_val = part_prob_maps.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        val_range = torch.clamp(max_val-min_val, min=1e-10)
        part_prob_maps = (part_prob_maps - min_val) / val_range * 255.

        prob_maps = part_prob_maps
        assert prob_maps.size(0) == len(original_ims)

        result_images = []
        for ii in range(raw_batch_images.size(0)):
            single_image = [raw_batch_images[ii]]  # 3HW
            part_probs = prob_maps[ii].split(dim=0, split_size=1)  # List[Tensor(1HW)]
            part_probs = [item.expand_as(batch_images[ii]) for item in part_probs]  # List[Tensor(3HW)]
            single_image += part_probs
            result_images += single_image

        grid = utils.make_grid(result_images, nrow=prob_maps.size(1) + 1, padding=2, normalize=False,
                               range=None, scale_each=False, pad_value=255)
        ndarr = grid.mul(1).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        filename = os.path.join(save_dir, "multilevel_probmap", '{}.jpg'.format(pid))
        im.save(filename)
    return

def plot_prob_maps_bak(model, image_items, in_size, save_dir=None):
    model.eval()
    MAX_NUM_ID = 5
    MAX_IMG_PER_ID = 10

    # collect image paths of all pids
    pid_to_fpath = defaultdict(list)
    for i, item in enumerate(image_items):
        # item(tuple): (img_path, pid, camid), pid = dataset_name + "_" + str(pid)
        pid = item[1]
        pid_to_fpath[pid].append(item[0])

    # Go through person identities
    for ii, (pid, fpath) in enumerate(pid_to_fpath.items()):
        if MAX_NUM_ID > 0 and ii >= MAX_NUM_ID:
            break

        print('Processing images of {} ({}/{}) ...'.format(pid, ii + 1, len(pid_to_fpath)))
        # list of cv2 images of a pid
        original_ims = [cv2.imread(x, cv2.IMREAD_COLOR) for x in fpath]
        # resize, to_tensor, normalize cv2im and concatenate as an input batch
        batch_images = torch.cat(
            [cv2im_transformer(x, im_size=in_size, norm=False, div=False) for x in original_ims],
            dim=0
        )
        batch_images = batch_images.cuda()  # BCHW
        raw_batch_images = batch_images.clone()

        #
        if batch_images.size(0) >= MAX_IMG_PER_ID:
            batch_images = batch_images[:MAX_IMG_PER_ID, ...]
            raw_batch_images = raw_batch_images[:MAX_IMG_PER_ID, ...]
            fpath = fpath[:MAX_IMG_PER_ID]

        # forward at eval mode
        model_input = {
            "images": batch_images,
            "img_paths": fpath,
            "flipped": [False] * len(fpath),
            "erased": [torch.tensor([-1, -1])] * len(fpath)
        }
        output = model(model_input)

        # (b num_parts h w) to (b num_parts H W)
        prob_maps = F.interpolate(output['part_maps'], size=batch_images.size()[-2:],
                                  mode='bilinear', align_corners=False)
        part_prob_maps = prob_maps[:, 1:, ...]
        
        _b, _num_part, _h = part_prob_maps.size()[:3]
        part_prob_maps = F.softmax(part_prob_maps.view(_b,_num_part, -1)*8.,dim=-1).view(_b, _num_part, _h, -1)

        #p_mask = part_prob_maps.max(dim=2, keepdim=True)[0].max(dim=3,keepdim=True)[0]  # (b num_parts 1 w)
        #part_prob_maps = part_prob_maps/p_mask * 255.

        max_val = part_prob_maps.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        min_val = part_prob_maps.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        val_range = torch.clamp(max_val-min_val, min=1e-10)
        part_prob_maps = (part_prob_maps - min_val) / val_range * 255.

        
        fg_prob_map = F.interpolate(output['fg_map'], size=batch_images.size()[-2:],
                                    mode='bilinear', align_corners=False)
        #fg_prob_map = fg_prob_map / fg_prob_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] * 255.
        fg_prob_map = fg_prob_map * 255.
        prob_maps = torch.cat([fg_prob_map, part_prob_maps[:, 0:, ...]], dim=1)
        assert raw_batch_images.size(0) == prob_maps.size(0)

        result_images = []
        for ii in range(raw_batch_images.size(0)):
            single_image = [raw_batch_images[ii]]  # 3HW
            part_probs = prob_maps[ii].split(dim=0, split_size=1)  # List[Tensor(1HW)]
            part_probs = [item.expand_as(batch_images[ii]) for item in part_probs]  # List[Tensor(3HW)]
            single_image += part_probs
            result_images += single_image

        grid = utils.make_grid(result_images, nrow=prob_maps.size(1) + 1, padding=2, normalize=False,
                               range=None, scale_each=False, pad_value=255)
        ndarr = grid.mul(1).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        filename = os.path.join(save_dir, 'prob_{}.jpg'.format(pid))
        im.save(filename)
    return

def save_fg_maps(model, image_items, in_size, save_dir=None):
    model.eval()
    #image_items = image_items[:13]
    TOT_NUM_IMG = len(image_items)
    MAX_BATCH_SIZE = 50
    MAX_NUM_BATCH = len(image_items) // MAX_BATCH_SIZE + 1

    cnt = 0
    while cnt < MAX_NUM_BATCH:
        print('Processing batch of {}/{} ...'.format(cnt, MAX_NUM_BATCH))
        END = (cnt + 1) * MAX_BATCH_SIZE if (cnt + 1) * MAX_BATCH_SIZE < TOT_NUM_IMG else TOT_NUM_IMG
        cur_batch = image_items[cnt * MAX_BATCH_SIZE:END]
        fpath = [item[0] for item in cur_batch]
        # list of cv2 images of a pid
        original_ims = [cv2.imread(x, cv2.IMREAD_COLOR) for x in fpath]
        # resize, to_tensor, normalize cv2im and concatenate as an input batch
        batch_images = torch.cat(
            [cv2im_transformer(x, im_size=in_size, norm=False, div=False) for x in original_ims],
            dim=0
        )
        batch_images = batch_images.cuda()  # BCHW
        raw_batch_images = batch_images.clone()
        # forward at eval mode
        model_input = {
            "images": batch_images,
            "img_paths": fpath,
            "flipped": [False] * len(fpath),
            "erased": [torch.tensor([-1, -1])] * len(fpath)
        }
        output = model(model_input)

        # (b 1 h w) to (b 1 H W)
        prob_maps = F.interpolate(output['prob_maps'], size=batch_images.size()[-2:],
                                  mode='bilinear', align_corners=False)
        assert raw_batch_images.size(0) == prob_maps.size(0) == len(fpath)
        # opt1
        max_val = prob_maps.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        min_val = prob_maps.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        prob_maps = (prob_maps - min_val) / (max_val - min_val)

        # opt2
        #prob_maps[prob_maps>=0.5] = 1
        #prob_maps[prob_maps<0.5] = 0
        
        prob_maps = prob_maps * 255.

        for ii, (name, fg) in enumerate(zip(fpath, prob_maps)):
            name = os.path.basename(name)
            fg = fg.expand_as(batch_images[ii])  # Tensor(3HW)
            ndarr = fg.mul(1).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(ndarr)
            filename = os.path.join(save_dir, name)
            im.save(filename)
        cnt += 1
    return

def plot_fg_maps(model, image_items, in_size, save_dir=None):
    model.eval()
    MAX_NUM_ID = 10
    MAX_BATCH_SIZE = 10

    # collect image paths of all pids
    pid_to_fpath = defaultdict(list)
    for i, item in enumerate(image_items):
        # item(tuple): (img_path, pid, camid), pid = dataset_name + "_" + str(pid)
        pid = item[1]
        pid_to_fpath[pid].append(item[0])

    # Go through person identities
    for ii, (pid, fpath) in enumerate(pid_to_fpath.items()):
        if MAX_NUM_ID > 0 and ii >= MAX_NUM_ID:
            break

        print('Processing images of {} ({}/{}) ...'.format(pid, ii + 1, len(pid_to_fpath)))
        # list of cv2 images of a pid
        original_ims = [cv2.imread(x, cv2.IMREAD_COLOR) for x in fpath]
        # resize, to_tensor, normalize cv2im and concatenate as an input batch
        batch_images = torch.cat(
            [cv2im_transformer(x, im_size=in_size, norm=False, div=False) for x in original_ims],
            dim=0
        )
        batch_images = batch_images.cuda()  # BCHW
        raw_batch_images = batch_images.clone()

        #
        if batch_images.size(0) >= MAX_BATCH_SIZE:
            batch_images = batch_images[:MAX_BATCH_SIZE, ...]
            raw_batch_images = raw_batch_images[:MAX_BATCH_SIZE, ...]
            fpath = fpath[:MAX_BATCH_SIZE]

        # forward at eval mode
        model_input = {
            "images": batch_images,
            "img_paths": fpath,
            "flipped": [False] * len(fpath),
            "erased": [torch.tensor([-1, -1])] * len(fpath)
        }
        output = model(model_input)

        # (b num_parts h w) to (b num_parts H W)
        # import pdb; pdb.set_trace()
        prob_maps = F.interpolate(output['prob_maps'], size=batch_images.size()[-2:],
                                  mode='bilinear', align_corners=False)

        fg_prob_map = prob_maps
        fg_prob_map = fg_prob_map / fg_prob_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] * 255.
        prob_maps = fg_prob_map
        assert raw_batch_images.size(0) == prob_maps.size(0)

        result_images = []
        for ii in range(raw_batch_images.size(0)):
            single_image = [raw_batch_images[ii]]  # 3HW
            part_probs = prob_maps[ii]  # Tensor(1HW)
            part_probs = part_probs.expand_as(batch_images[ii])  # Tensor(3HW)
            single_image += [part_probs]
            result_images += single_image

        grid = utils.make_grid(result_images, nrow=prob_maps.size(1) + 1, padding=2, normalize=False,
                               range=None, scale_each=False, pad_value=255)
        ndarr = grid.mul(1).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        filename = os.path.join(save_dir, 'fg_{}.jpg'.format(pid))
        im.save(filename)
    return

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    elif args.vis_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False

        train_items, pid_to_lbl, cfg = DefaultTrainer.build_vis_loader(cfg)
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        DefaultTrainer.prob_maps(cfg, train_items, pid_to_lbl, model)
        return

    trainer = DefaultTrainer(cfg)
    if args.finetune:
        Checkpointer(trainer.model).load(cfg.MODEL.WEIGHTS)  # load trained model to fine-tune

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

def vis():
    import shutil
    bsl_dir = '/home/caffe/code/fastReID/logs/OccDuke/ranking-list-bsl/'
    ours_dir = '/home/caffe/code/fastReID/logs/OccDuke/ranking-list-ours/'
    dst_dir = '/home/caffe/code/fastReID/logs/OccDuke/ranking-list-dst/'
    bsl_imgs = os.listdir(bsl_dir)
    ours_imgs = os.listdir(ours_dir)
    for img in ours_imgs:
        if img in bsl_imgs:
            dst_name = img.split('.')[0] + "_ours.png"
            shutil.copy(os.path.join(ours_dir, img), os.path.join(dst_dir, dst_name))
            dst_name = img.split('.')[0] + "_bsl.png"
            shutil.copy(os.path.join(bsl_dir, img), os.path.join(dst_dir, dst_name))
    return

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    # vis()
