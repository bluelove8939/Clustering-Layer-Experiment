import os

import torch
from torch.utils.data import DataLoader

import Resnet50_STL10_normal as normal
import Resnet50_STL10_clustered as clustered

from custom_layers.clustering import ClusteredModelOptimizer


# Testbench settings
resultfile_name = 'resnet_optimization_result.log'
resultfile_path = os.path.join(os.curdir, 'logs', resultfile_name)
clust_layer_num = 5
thresholds = [0.1, 0.08, 0.05, 0.02, 0.01, 0.005]
clustering_test_results = []

# Test dataset generation
total_datasize = len(clustered.test_dataset)
valid_datasize = int(total_datasize / 16)
print(f"test dataset size: {valid_datasize}/{total_datasize}")
test_dataset, _ = torch.utils.data.random_split(clustered.test_dataset, [valid_datasize, total_datasize-valid_datasize])
test_loader = DataLoader(test_dataset, batch_size=10)
loss_fn = clustered.loss_fn

# Test with normal model as a reference
model = normal.model
model.load_state_dict(torch.load(normal.save_fullpath))

print('testing with normal model')
accuracy, avg_loss = clustered.test(test_loader, model, loss_fn)

# Set optimizer module
opt_module = ClusteredModelOptimizer(test_loader, loss_fn=loss_fn)
model = clustered.model
model.load_state_dict(torch.load(normal.save_fullpath))
opt_module.optimize_model(model, target_acc=accuracy, thres_max=1, thres_min=0.001, max_iter=10, verbose=1)

# Print test results
clust_acc, clust_avg_loss = clustered.test(test_loader, model, loss_fn)
print(f"target accuracy: {accuracy}  target avereage loss: {avg_loss:.4f}")
print(f"clust accuracy:  {clust_acc}  clust average loss:   {clust_avg_loss:.4f}")
print(f"selected thresholds: {', '.join(list(map(lambda x: f'{x.threshold:.4f}', model.clust_layers)))}")
