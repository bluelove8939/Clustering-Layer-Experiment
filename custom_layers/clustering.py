import os
import logging
import datetime

import torch
import torch.nn as nn

from tools.training import test
from tools.progressbar import progressbar


if 'logs' not in os.listdir():
    os.mkdir(os.path.join(os.curdir, 'logs'))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] %(message)s')


def set_logger(info=None, verbose=False):
    global logger

    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logfile_path = os.path.join(os.curdir, 'logs', f'cluster_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    if info is not None:
        logfile_path = os.path.join(os.curdir, 'logs', f'cluster_{info}.log')
    logfile_handler = logging.FileHandler(logfile_path)
    logfile_handler.setFormatter(formatter)
    logger.addHandler(logfile_handler)

    return logger


# @torch.fx.wrap
# def cluster_img(img, threshold, cacheline_size):
#     img_shape = img.size()
#     img_reshaped = img.reshape(-1, cacheline_size)
#     clustered_masks = torch.zeros_like(img_reshaped).type(torch.bool)
#     masks = torch.ones_like(img_reshaped)
#
#     for idx in range(cacheline_size):
#         sel_column = img_reshaped[:, idx]
#         masks[:, idx] = 0
#         # masks_embed = masks[:, idx]
#         # masks_embed -= masks_embed
#         sel_column = sel_column.repeat(cacheline_size, 1).transpose(0, 1)
#         delta_masks = (torch.abs(img_reshaped - sel_column).le(threshold) * masks).type(torch.bool)
#         img_reshaped = torch.where(torch.logical_and(delta_masks == True, clustered_masks == False),
#                                    sel_column, img_reshaped)
#         clustered_masks = torch.logical_or(clustered_masks, delta_masks)
#
#     return torch.reshape(img_reshaped, img_shape)


@torch.fx.wrap
def cluster_img(img, threshold, cacheline_size):
    img_shape = img.size()
    img_reshaped = img.reshape(-1, cacheline_size)
    centroids = img_reshaped[:, 0].clone().detach()
    # counts = torch.zeros_like(centroids)

    for idx in range(cacheline_size):
        sel_column = img_reshaped[:, idx]
        mask = torch.le(torch.abs(sel_column - centroids), threshold)
        sel_column[:] = torch.where(mask, centroids, sel_column)
        centroids[:] = torch.where(mask, centroids, sel_column)
        # counts[:] = torch.where(mask, counts+1, counts)

    return torch.reshape(img_reshaped, img_shape)


class ClusteringFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, cacheline_size):
        x_clustered = cluster_img(x, threshold, cacheline_size)
        return x_clustered

    @staticmethod
    def backward(ctx, *grad_outputs):
        return *grad_outputs, None, None


class ClusteringLayer(nn.Module):
    def __init__(self, threshold=0, cacheline_size=64):
        super(ClusteringLayer, self).__init__()
        self.threshold = threshold
        self.cacheline_size = cacheline_size

        self._clust_amt_psum = 0
        self._clust_base_cnt = 0
        self._clust_cnt = 0

    def forward(self, x):
        x_copied = x.clone().detach()
        logger.debug("clustering forward method called")
        y = ClusteringFunction.apply(x, self.threshold, self.cacheline_size)
        cmp = torch.ne(x_copied, y)
        self._clust_amt_psum += torch.count_nonzero(cmp).item() / torch.numel(y)
        self._clust_base_cnt += torch.numel(torch.unique(y)) / torch.numel(y)
        self._clust_cnt += 1
        return y

    def reset_clust_amt(self):
        self._clust_amt_psum = 0
        self._clust_base_cnt = 0
        self._clust_cnt = 0

    def get_clust_amt(self):
        return self._clust_amt_psum / self._clust_cnt

    def get_clust_base_cnt(self):
        return self._clust_base_cnt / self._clust_cnt

    def string(self):
        return "Custom Layer: Clustering"


class QuantClusteringLayer(nn.Module):
    def __init__(self, threshold=0, cacheline_size=64):
        super(QuantClusteringLayer, self).__init__()
        self.threshold = threshold
        self.cacheline_size = cacheline_size

    def forward(self, x):
        return ClusteringFunction.apply(x, self.threshold, self.cacheline_size)

    def string(self):
        return "Custom Layer: Clustering"


class ClusteredModelOptimizer(object):
    def __init__(self, tuning_dataloader, loss_fn):
        self.tuning_dataloader = tuning_dataloader
        self.loss_fn = loss_fn

    def optimize_model(self, model, target_acc=100, thres_max=1, thres_min=0.001, max_iter=10, verbose=1):
        print("\nOptimization Config")
        print(f"- target_acc: {target_acc}")
        print(f"- thres_max: {thres_max}")
        print(f"- thres_min: {thres_min}")
        print(f"- max_iter: {max_iter}")

        thres = [thres_min] * len(model.clust_layers)

        for tidx in reversed(range(len(thres))):
            tmin = thres_min
            tmax = thres_max

            for sidx in range(max_iter):
                tcen = (tmax + tmin) / 2
                thres[tidx] = tcen
                model.set_clust_threshold(*thres)
                model.reset_clust_layer()

                if verbose:
                    print(f"selecting thres[{tidx}] = {tcen}")

                acc, _ = test(self.tuning_dataloader, model, loss_fn=self.loss_fn, verbose=1)

                if verbose:
                    print(f"clust amt: {', '.join(list(map(lambda x: f'{x:.4f}', model.get_clust_amt())))}")
                    print(f"base cnt: {', '.join(list(map(lambda x: f'{x:.4f}', model.get_clust_base_cnt())))}")

                if acc >= target_acc:
                    tmin = tcen
                else:
                    tmax = tcen

            if verbose:
                print(f"\nselected threshold[{tidx}] = {thres[tidx]}\n")


# if __name__ == '__main__':
#     img = torch.tensor(
#         [
#             [1, 3, 2, 4],
#             [2, 2, 1, 2],
#             [3, 2, 4, 4],
#             [2, 3, 1, 1],
#         ]
#     )
#
#     print(img)
#     print(cluster_img_updated(img, 1, 8))
