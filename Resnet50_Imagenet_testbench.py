import os
from itertools import product

import torch
from torch.utils.data import DataLoader

import Resnet50_Imagenet_normal as normal
import Resnet50_Imagenet_clustered as clustered

from custom_layers.clustering import ClusteredModelOptimizer
from tools.imagenet_utils.args_generator import args
from tools.imagenet_utils.training import train, validate


# Testbench settings
resultfile_name = 'resnet_imagenet_testbench_result.csv'
resultfile_path = os.path.join(os.curdir, 'logs', resultfile_name)
clust_layer_num = 5
thresholds = [0.1, 0.08, 0.05, 0.02, 0.01, 0.005]
clustering_test_results = []

# Test dataset generation
# total_datasize = len(clustered.test_dataset)
# valid_datasize = int(total_datasize / 100)
# print(f"test dataset size: {valid_datasize}/{total_datasize}")
# test_dataset, _ = torch.utils.data.random_split(clustered.train_dataset, [valid_datasize, total_datasize-valid_datasize])
# test_loader = DataLoader(test_dataset, batch_size=10)
test_loader = clustered.test_loader
loss_fn = clustered.loss_fn

# Test with normal model as a reference
model = normal.model
model.load_state_dict(torch.load(normal.save_fullpath))

print('testing with normal model')
accuracy, avg_loss = validate(test_loader, model, loss_fn, args)

clustering_test_results.append(','.join(map(str, [
    *([0]*clust_layer_num), accuracy, avg_loss, *([0, 0]*clust_layer_num),
])))


# Testcase generator
def test_case_gen(layer_num, **kwargs):
    testcases = []

    if "uniform" in kwargs.keys() and kwargs.get('uniform') == True:
        for thres in thresholds:
            testcases.append([thres] * layer_num)
        return testcases

    for case in product(*[thresholds for _ in range(layer_num)]):
        flag = False
        for idx in range(layer_num):
            if f"layer{idx}_min" in kwargs.keys() and case[idx] < kwargs.get(f"layer{idx}_min"): flag = True
            if f"layer{idx}_max" in kwargs.keys() and case[idx] < kwargs.get(f"layer{idx}_max"): flag = True

        if not flag:
            testcases.append(case)
    return testcases


# Main testbench
if __name__ == '__main__':
    for testcase in test_case_gen(clust_layer_num, uniform=False):
        # clustered.clustering.set_logger(info=f"{testcase}", verbose=False)

        model = clustered.model
        model.load_state_dict(torch.load(normal.save_fullpath))
        model.reset_clust_layer()
        model.set_clust_threshold(*testcase)

        print(f'testing with threshold value: {testcase}')
        accuracy, avg_loss = validate(test_loader, model, loss_fn, args)

        clustering_test_results.append(','.join(map(str, [
            *testcase, accuracy, avg_loss, *model.get_clust_amt(), *model.get_clust_base_cnt(),
        ])))

        with open(resultfile_path, 'wt') as file:
            file.write('\n'.join(clustering_test_results))
