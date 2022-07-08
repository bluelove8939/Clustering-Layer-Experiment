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

def cluster_img(img, threshold, cacheline_size):
    img_shape = img.shape
    img_flatten = torch.flatten(img)
    pivot = 0
    clust_cnt = 0
    length = img_flatten.shape[-1]

    while pivot + cacheline_size - 1 < length:
        bases = []
        for pidx in range(pivot, pivot + cacheline_size):
            flag = False
            for idx, val in enumerate(bases):
                if abs(val - img_flatten[pidx]) < threshold:
                    img_flatten[pidx], flag = val, True
                    clust_cnt += 1
                    break
            if not flag:
                bases.append(img_flatten[pidx])

        pivot += cacheline_size

    logger.debug(f"clustered result: {clust_cnt}/{length}")

    return torch.reshape(img_flatten, img_shape)


class ClusteringFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, dilation, padding, stride, threshold, cacheline_size):
        x_clustered = cluster_img(x, threshold, cacheline_size)
        return x_clustered

    @staticmethod
    def backward(ctx, *grad_outputs):
        return *grad_outputs, None, None, None, None, None, None, None


class ClusteringLayer(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1, threshold=0, cacheline_size=64):
        super(ClusteringLayer, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.threshold = threshold
        self.cacheline_size = cacheline_size

        self._clust_amt_psum = 0
        self._clust_cnt = 0

    def forward(self, x):
        logger.debug("clustering forward method called")
        y = ClusteringFunction.apply(x,self.kernel_size, self.dilation, self.padding, self.stride,
                                     self.threshold, self.cacheline_size)
        cmp = torch.eq(x, y)
        self._clust_amt_psum += torch.numel(cmp) - torch.count_nonzero(cmp)
        self._clust_cnt += 1
        return y

    def reset_clust_amt(self):
        self._clust_amt_psum = 0
        self._clust_cnt = 0

    def get_clust_amt(self):
        return self._clust_amt_psum / self._clust_cnt

    def string(self):
        return "Custom Layer: Clustering"
