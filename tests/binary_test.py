import cv2
import numpy as np
from skimage import filters
import torch

def blur_and_threshold(masks):
    """
    Args:
        masks(4D tensor): (b 1 h w)
    """
    device = masks.device
    masks = masks.split(dim=0, split_size=1)
    masks = [m.squeeze().unsqueeze(-1).cpu().numpy() for m in masks]  # (h w 1)
    post_masks = []
    for m in masks:
        # blur = cv2.GaussianBlur(m, ksize=(5, 5), sigmaX=0)  # (h w)
        blur = m[:, :, 0]
        # thr = filters.threshold_otsu(blur)
        # thr = np.mean(blur)
        thr = blur.ravel()[blur.ravel().argsort()[64]]
        mask = torch.from_numpy(blur > thr).float()
        post_masks.append(mask)
    post_masks = torch.stack(post_masks, dim=0)[:, None, :, :]  # (b 1 h w)
    post_masks.to(device)
    return post_masks

if __name__ == '__main__':
    masks = torch.rand(2, 1, 24, 8).cuda()
    out = blur_and_threshold(masks)
    print(out.shape)
