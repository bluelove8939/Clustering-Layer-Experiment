import os
from itertools import product

import torch
from torch.utils.data import DataLoader

import CustomCNN_MNIST_normal as normal
import CustomCNN_MNIST_clustered as clustered


# Testbench settings
resultfile_name = 'clustering_test_results_diverse.csv'
resultfile_path = os.path.join(os.curdir, 'logs', resultfile_name)
clust_layer_num = 4
thresholds = [1, 0.5, 0.1]
clustering_test_results = []

# Test dataset generation
total_datasize = len(clustered.test_dataset)
valid_datasize = int(total_datasize * 0.1)
test_dataset, _ = torch.utils.data.random_split(clustered.test_dataset, [valid_datasize, total_datasize-valid_datasize])
test_loader = DataLoader(test_dataset, batch_size=5)
loss_fn = clustered.loss_fn

# Test with normal model as a reference
model = normal.NetworkModel()
model.load_state_dict(torch.load(normal.save_fullpath))

print('testing with normal model')
accuracy, avg_loss = clustered.test(test_loader, model, loss_fn)

clustering_test_results.append(','.join(map(str, [
    *([0]*clust_layer_num), accuracy, avg_loss, *([0]*clust_layer_num)
])))


# Testcase generator
def test_case_gen(layer_num, **kwargs):
    testcases = []
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
    for testcase in test_case_gen(clust_layer_num):
        # clustered.clustering.set_logger(info=f"{testcase}", verbose=False)

        model = clustered.NetworkModel()
        model.load_state_dict(torch.load(normal.save_fullpath))
        model.reset_clust_layer()
        model.set_clust_threshold(*testcase)

        print(f'testing with threshold value: {testcase}')
        accuracy, avg_loss = clustered.test(test_loader, model, loss_fn)

        clustering_test_results.append(','.join(map(str, [
            *testcase, accuracy, avg_loss, *model.get_clust_amt(),
        ])))

        with open(resultfile_path, 'wt') as file:
            file.write('\n'.join(clustering_test_results))