# credits: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank.py
import os
import cv2
from PIL import Image

import warnings
from collections import defaultdict

import faiss
import numpy as np

try:
    from .rank_cylib.rank_cy import evaluate_cy

    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython rank evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )

def eval_cuhk03(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10

    num_q, num_g = distmat.shape
    dim = q_feats.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(g_feats)
    if use_distmat:
        indices = np.argsort(distmat, axis=1)
    else:
        _, indices = index.search(q_feats, k=num_g)

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
                format(num_g)
        )

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_market1501(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    dim = q_feats.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(g_feats)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    if use_distmat:
        indices = np.argsort(distmat, axis=1)
    else:
        _, indices = index.search(q_feats, k=num_g)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP

def evaluate_py(
        distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03, use_distmat
):
    if use_metric_cuhk03:
        return eval_cuhk03(
            distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat
        )
    else:
        return eval_market1501(
            distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank, use_distmat
        )

def evaluate_rank(
        distmat,
        q_feats,
        g_feats,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank=50,
        use_metric_cuhk03=False,
        use_distmat=False,
        use_cython=True
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_feats (numpy.ndarray): 2-D array containing query features.
        g_feats (numpy.ndarray): 2-D array containing gallery features.
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_cy(
            distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03, use_distmat
        )
    else:
        return evaluate_py(
            distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03, use_distmat
        )

def evaluate_rank_with_plot(
        distmat,
        q_feats,
        g_feats,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        q_paths,
        g_paths,
        max_rank=50,
        use_metric_cuhk03=False,
        use_distmat=False,
        use_cython=True
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_feats (numpy.ndarray): 2-D array containing query features.
        g_feats (numpy.ndarray): 2-D array containing gallery features.
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """

    def read_im(im_path):
        # shape [H, W, 3]
        im = np.asarray(Image.open(im_path))
        # Resize to (im_h, im_w) = (128, 64)
        resize_h_w = (128, 64)
        if (im.shape[0], im.shape[1]) != resize_h_w:
            im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
        # shape [3, H, W]
        im = im.transpose(2, 0, 1)
        return im

    def add_border(im, border_width, value):
        """Add color border around an image. The resulting image size is not changed.
        Args:
          im: numpy array with shape [3, im_h, im_w]
          border_width: scalar, measured in pixel
          value: scalar, or numpy array with shape [3]; the color of the border
        Returns:
          im: numpy array with shape [3, im_h, im_w]
        """
        assert (im.ndim == 3) and (im.shape[0] == 3)
        im = np.copy(im)

        if isinstance(value, np.ndarray):
            # reshape to [3, 1, 1]
            value = value.flatten()[:, np.newaxis, np.newaxis]
        im[:, :border_width, :] = value
        im[:, -border_width:, :] = value
        im[:, :, :border_width] = value
        im[:, :, -border_width:] = value

        return im

    def make_im_grid(ims, space, pad_val):
        """Make a grid of images with space in between.
        Args:
          ims: a list of [3, im_h, im_w] images
          space: the num of pixels between two images
          pad_val: scalar, or numpy array with shape [3]; the color of the space
        Returns:
          ret_im: a numpy array with shape [3, H, W]
        """
        n_cols = len(ims)
        k_space = 5  # k_space means q_g space
        assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
        h, w = ims[0].shape[1:]
        H = h
        W = w * n_cols + space * (n_cols - 2) + k_space * space
        if isinstance(pad_val, np.ndarray):
            # reshape to [3, 1, 1]
            pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
        ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)

        ret_im[:, 0:h, 0:w] = ims[0]  # query image

        start_w = w + k_space * space
        for im in ims[1:]:
            end_w = start_w + w
            ret_im[:, 0:h, start_w:end_w] = im
            start_w = end_w + space
        return ret_im

    def save_rank_result(query, top_gallery, save_path):
        """Save a query and its rank list as an image.
        Args:
            query: query image path
            top_gallery (list): top gallery image paths
            save_path:
        """
        query_id = int(os.path.basename(query).split('_')[0])
        top10_ids = [int(os.path.basename(p).split('_')[0]) for p in top_gallery]

        images = [read_im(query)]

        for gallery_path, gallery_id in zip(top_gallery, top10_ids):
            g_im = read_im(gallery_path)
            # Add green boundary to true positive, red to false positive
            color = np.array([0, 255, 0]) if query_id == gallery_id else np.array([255, 0, 0])
            g_im = add_border(g_im, 3, color)
            images.append(g_im)

        im = make_im_grid(images, space=4, pad_val=255)
        im = im.transpose(1, 2, 0)
        Image.fromarray(im).save(save_path)

    num_q, num_g = distmat.shape
    dim = q_feats.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(g_feats)

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    if use_distmat:
        indices = np.argsort(distmat, axis=1)
    else:
        _, indices = index.search(q_feats, k=num_g)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # --- plot ranking results ----------------------------------------
        rank_result_dir = "/home/caffe/code/fastReID/logs/OccDuke/ranking-list-bsl"
        plot_ranking = True
        if plot_ranking:
            assert q_paths is not None and g_paths is not None
            g_paths = np.asarray(g_paths)
            idx = np.where(keep == 1)[0]
            top10 = g_paths[indices[q_idx]][idx][:10].tolist()
            top10_ids = g_pids[indices[q_idx]][idx][:10].tolist()

            flag = top10_ids[0] != q_pids[q_idx]  # only plot ranking list of error top1
            # flag = top10_ids[0] == q_pids[q_idx] # only plot ranking list of correct top1
            # flag = True # plot ranking list of all queries
            if flag:
                print("Processing {}-th/{} image ...".format(q_idx, num_q))
                save_rank_result(q_paths[q_idx], top10, save_path=os.path.join(rank_result_dir, os.path.basename(q_paths[q_idx])))

                # save ground truth ranking list
                # ground_truth = ((g_pids[indices[q_idx]] == q_pids[q_idx]) &  # NOTE: same id but different camera
                #                 (g_camids[indices[q_idx]] != q_camids[q_idx]))
                # ground_truth = np.where(ground_truth == 1)[0]
                # top10 = g_paths[indices[q_idx]][ground_truth][:10].tolist()
                # save_rank_result(q_paths[q_idx], top10, save_path=os.path.join(rank_result_dir, os.path.basename(q_paths[q_idx]).split('.')[0] + '_gt.jpg'))
        # ---------------------------------------------------------------------

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP

def simple_evaluate_rank(distmat,
                         q_pids,
                         g_pids,
                         q_camids,
                         g_camids,
                         max_rank=50):
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP
