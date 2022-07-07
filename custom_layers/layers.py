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
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)
logfile_path = os.path.join(os.curdir, 'logs', f'cluster_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logfile_handler = logging.FileHandler(logfile_path)
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)


def cluster_img(img, threshold, cacheline_size):
    img_shape = img.shape
    img_flatten = torch.flatten(img)
    pivot = 0
    clust_cnt = 0
    length = img_flatten.shape[-1]

    while pivot + cacheline_size - 1 < length:
        # logger.debug(f"pivot: {pivot + cacheline_size:4d}/{length - (length % cacheline_size):4d}")

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
        fold_params = dict(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        ctx.save_for_backward(x)
        fold_layer = nn.Fold(output_size=x.shape[2:], **fold_params)
        unfold_layer = nn.Unfold(**fold_params)

        x_unfolded = unfold_layer(x)

        if len(x_unfolded.shape) == 2:
            x_clustered = cluster_img(x_unfolded, threshold, cacheline_size)
            return fold_layer(x_clustered)

        batchsiz = x_unfolded.shape[0]
        for bidx in range(batchsiz):
            batch_clustered = cluster_img(x_unfolded[bidx], threshold, cacheline_size)
            x_unfolded[bidx] = batch_clustered

        return fold_layer(x_unfolded)

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
