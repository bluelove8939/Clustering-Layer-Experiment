import os
import sys
import logging
import datetime

import torch
import torch.nn as nn


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


# # Deprecated cluster_img method
# def cluster_img(img, threshold, cacheline_size):
#     img_shape = img.shape
#     img_flatten = torch.flatten(img)
#     pivot = 0
#     clust_cnt = 0
#     length = img_flatten.shape[-1]
#
#     while pivot + cacheline_size - 1 < length:
#         bases = []
#         for pidx in range(pivot, pivot + cacheline_size):
#             flag = False
#             for val in bases:
#                 if abs(val - img_flatten[pidx]) < threshold:
#                     img_flatten[pidx], flag = val, True
#                     clust_cnt += 1
#                     break
#             if not flag:
#                 bases.append(img_flatten[pidx])
#
#         pivot += cacheline_size
#
#     logger.debug(f"clustered result: {clust_cnt}/{length}")
#
#     return torch.reshape(img_flatten, img_shape)


def cluster_img(img, threshold, cacheline_size):
    img_shape = img.shape
    img_reshaped = img.view(-1, cacheline_size)
    masks = torch.ones_like(img_reshaped)

    for idx in range(cacheline_size):
        sel_column = img_reshaped[:, idx]
        masks[:, idx] = 0
        sel_column = sel_column.repeat(cacheline_size, 1).transpose(0, 1)
        delta_masks = (torch.abs(img_reshaped - sel_column).le(threshold) * masks).type(torch.bool)
        img_reshaped = torch.where(delta_masks == True, sel_column, img_reshaped)

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
        self._clust_cnt = 0

    def forward(self, x):
        x_copied = x.clone().detach()
        logger.debug("clustering forward method called")
        y = ClusteringFunction.apply(x, self.threshold, self.cacheline_size)
        cmp = torch.ne(x_copied, y)
        self._clust_amt_psum += torch.count_nonzero(cmp).item() / torch.numel(y)
        self._clust_cnt += 1
        return y

    def reset_clust_amt(self):
        self._clust_amt_psum = 0
        self._clust_cnt = 0

    def get_clust_amt(self):
        return self._clust_amt_psum / self._clust_cnt

    def string(self):
        return "Custom Layer: Clustering"


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
