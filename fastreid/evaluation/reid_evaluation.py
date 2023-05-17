# encoding: utf-8
"""
# Last Change:  2022-08-07 17:33:10
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict
from sklearn import metrics

import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank, evaluate_rank_with_plot, simple_evaluate_rank
from .rerank import re_ranking
from .roc import evaluate_roc
from fastreid.utils import comm

logger = logging.getLogger(__name__)

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch tensor, with shape [m, d]
        y: pytorch tensor, with shape [n, d]
    Returns:
        dist: pytorch tensor, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def compute_dist(array1, array2, dist_type='cosine', normalize=True):
    """
    Args:
        array1: tensor, with shape [m, d]
        array2: tensor, with shape [n, d]
    Returns:
        dist: tensor, with shape [m, n]
    """
    if dist_type == 'cosine':
        if normalize:
            array1 = F.normalize(array1, dim=1)
            array2 = F.normalize(array2, dim=1)
        dist = - torch.mm(array1, array2.t())
        # Turn distance into positive value
        dist += 1
    elif dist_type == 'euclidean':
        dist = euclidean_dist(array1, array2)
    else:
        raise NotImplementedError
    return dist

def compute_dist_with_qg_visibility(array1,
                                    array2,
                                    vis1,
                                    vis2,
                                    dist_type='cosine',
                                    avg_by_vis_num=True):
    """Compute the euclidean or cosine distance of all pairs, considering part visibility.
    In this version, the distance of a <query part, gallery part> pair if only calculated when
    both are visible. And finally, distance of a <query image, gallery image> pair is set to a
    large value, if they do not have commonly visible part.
    Args:
        array1: numpy array with shape [m1, p, d]
        array2: numpy array with shape [m2, p, d]
        vis1: numpy array with shape [m1, p], p is num_parts
        vis2: numpy array with shape [m2, p], p is num_parts
        dist_type: one of ['cosine', 'euclidean']
        avg_by_vis_num: for each <query_image, gallery_image> distance, average the
            summed distance by the number of commonly visible parts.
    Returns:
        dist: numpy array with shape [m1, m2]
    """
    err_msg = "array1.shape = {}, vis1.shape = {}, array2.shape = {}, vis2.shape = {}" \
        .format(array1.shape, vis1.shape, array2.shape, vis2.shape)
    assert len(array1.shape) == 3, err_msg
    assert len(array2.shape) == 3, err_msg
    assert array1.shape[0] == vis1.shape[0], err_msg
    assert array2.shape[0] == vis2.shape[0], err_msg
    assert array1.shape[2] == array2.shape[2], err_msg
    assert array1.shape[1] == array2.shape[1] == vis1.shape[1] == vis2.shape[1], err_msg
    m1 = array1.shape[0]
    m2 = array2.shape[0]
    p = vis1.shape[1]
    d = array1.shape[2]
    dist = 0
    vis_sum = 0
    for i in range(p):
        # [m1, m2]
        dist_ = compute_dist(array1[:, i, :], array2[:, i, :], dist_type=dist_type)
        # opt1: hard
        # q_visible = vis1[:, i].unsqueeze(1).repeat([1, m2]) != 0
        # g_visible = vis2[:, i].unsqueeze(0).repeat([m1, 1]) != 0
        # visible = (q_visible & g_visible).float()

        # opt2: soft
        q_visible = vis1[:, i].unsqueeze(1).repeat([1, m2])
        g_visible = vis2[:, i].unsqueeze(0).repeat([m1, 1])
        visible = q_visible * g_visible  # (m1 m2)

        dist += dist_ * visible
        vis_sum += visible
    if avg_by_vis_num:
        dist /= vis_sum
    return dist, vis_sum

class ReIDEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs (Dict):
            outputs (Dict):
        """
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.img_paths.extend(inputs["img_paths"])
        self.features.append(outputs["feature"].cpu())

    @staticmethod
    def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            dist = 1 - torch.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.cpu().numpy()

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            pids = self.pids
            camids = self.camids
            img_paths = self.img_paths

        features = torch.cat(features, dim=0)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])
        query_img_paths = img_paths[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])
        gallery_img_paths = img_paths[self._num_query:]

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            re_dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = evaluate_rank(re_dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids,
                                                 gallery_camids, use_distmat=True)
        else:
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = evaluate_rank(dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids, gallery_camids,
                                                 use_distmat=False)
            # cmc, all_AP, all_INP = evaluate_rank_with_plot(
            #     dist, query_features, gallery_features,
            #     query_pids, gallery_pids, query_camids, gallery_camids,
            #     query_img_paths, gallery_img_paths,
            #     use_distmat=False
            # )

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_features, gallery_features,
                                          query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

class PartialReIDEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []
        self.img_paths = []
        self.p_visibility = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs (Dict):
            outputs (Dict):
        """
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.img_paths.extend(inputs["img_paths"])
        self.features.append(outputs["feature"].cpu())
        self.p_visibility.append(outputs["visibility"].cpu())

    @staticmethod
    def cal_dist_(metric: str,
                  query_feat: torch.tensor,
                  gallery_feat: torch.tensor,
                  q_p_vis: torch.tensor,
                  g_p_vis: torch.tensor):
        """
        Args:
            metric:
            query_feat (3D Tensor): (m p+2 d) where m: query number, p: part number, d: dim of feature
            gallery_feat (3D Tensor): (n p+2 d)
            q_p_vis (2D Tensor): (m p)
            g_p_vis (2D Tensor): (n p)
        """
        assert metric in ["holistic", "partial"], "must choose from [holistic, partial], but got {}".format(metric)
        if metric == "holistic":
            query_feat = F.normalize(query_feat, dim=-1)
            query_feat = query_feat.view(query_feat.size(0), -1)  # (m p+2*d)
            gallery_feat = F.normalize(gallery_feat, dim=-1)
            gallery_feat = gallery_feat.view(gallery_feat.size(0), -1)  # (n p+2*d)

            dist = compute_dist(query_feat, gallery_feat)
        else:
            num_parts = q_p_vis.size(1)
            num_holistic = query_feat.size(1) - num_parts
            q_h_feat = query_feat[:, :num_holistic, :]
            q_p_feat = query_feat[:, num_holistic:, :]
            g_h_feat = gallery_feat[:, :num_holistic, :]
            g_p_feat = gallery_feat[:, num_holistic:, :]

            ### holistic distance
            # q_h_feat = F.normalize(q_h_feat, dim=-1).view(q_h_feat.size(0), -1)
            # g_h_feat = F.normalize(g_h_feat, dim=-1).view(g_h_feat.size(0), -1)
            # h_dist = compute_dist(q_h_feat, g_h_feat)

            ### part distance
            dist, vis_sum = compute_dist_with_qg_visibility(q_p_feat, g_p_feat,
                                                            q_p_vis, g_p_vis, avg_by_vis_num=False)
            # dist += h_dist
            # vis_sum += 2

            dist /= vis_sum

        return dist.cpu().numpy()

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            pids = self.pids
            camids = self.camids
            img_paths = self.img_paths
            p_visibility = self.p_visibility

        features = torch.cat(features, dim=0)  # (N p dim_feat)
        p_visibility = torch.cat(p_visibility, dim=0)  # (N p)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])
        query_img_paths = img_paths[:self._num_query]
        query_p_vis = p_visibility[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])
        gallery_img_paths = img_paths[self._num_query:]
        gallery_p_vis = p_visibility[self._num_query:]

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = self.cal_dist_(self.cfg.TEST.METRIC, query_features, gallery_features,
                              query_p_vis, gallery_p_vis)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            re_dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = simple_evaluate_rank(re_dist,
                                                        query_pids, gallery_pids,
                                                        query_camids, gallery_camids)
        else:
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = simple_evaluate_rank(dist,
                                                        query_pids, gallery_pids,
                                                        query_camids, gallery_camids)
            # cmc, all_AP, all_INP = evaluate_rank_with_plot(
            #     dist, query_features, gallery_features,
            #     query_pids, gallery_pids, query_camids, gallery_camids,
            #     query_img_paths, gallery_img_paths,
            #     use_distmat=False
            # )

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_features, gallery_features,
                                          query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)
