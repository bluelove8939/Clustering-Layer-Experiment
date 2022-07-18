import os

import torch
from torch.utils.data import DataLoader

import Resnet50_Imagenet_normal as normal
import Resnet50_Imagenet_clustered as clustered

from custom_layers.clustering import ClusteredModelOptimizer
from tools.imagenet_utils.args_generator import args
from tools.imagenet_utils.training import train, validate


# Testbench settings
resultfile_name = 'resnet_imagenet_optimization_result.log'
resultfile_path = os.path.join(os.curdir, 'logs', resultfile_name)
clust_layer_num = 5
clustering_test_results = []

# Test dataset generation
total_datasize = len(clustered.train_dataset)
valid_datasize = int(total_datasize / 50)
print(f"test dataset size: {valid_datasize}/{total_datasize}")
tuning_dataset, _ = torch.utils.data.random_split(clustered.train_dataset, [valid_datasize, total_datasize-valid_datasize])
tuning_loader = DataLoader(tuning_dataset, batch_size=128)
loss_fn = clustered.loss_fn

# Test with normal model as a reference
model = normal.model
model.load_state_dict(torch.load(normal.save_fullpath))

print('testing with normal model')
accuracy, avg_loss = validate(tuning_loader, model, loss_fn, args)

# Set optimizer module
opt_module = ClusteredModelOptimizer(tuning_loader, loss_fn=loss_fn)
model = clustered.model
model.load_state_dict(torch.load(normal.save_fullpath))
opt_module.optimize_model(model, target_acc=accuracy, thres_max=1, thres_min=0.001, max_iter=20, verbose=1)

# Print test results and save result as text file
clust_acc, clust_avg_loss = validate(tuning_loader, model, loss_fn, args)
print(f"target accuracy: {accuracy}  target avereage loss: {avg_loss:.4f}")
print(f"clust accuracy:  {clust_acc}  clust average loss:   {clust_avg_loss:.4f}")
print(f"selected thresholds: {', '.join(list(map(lambda x: f'{x.threshold:.4f}', model.clust_layers)))}")

with open(resultfile_path, 'wt') as file:
    file.writelines([
        f"target accuracy: {accuracy}  target avereage loss: {avg_loss:.4f}\n",
        f"clust accuracy:  {clust_acc}  clust average loss:   {clust_avg_loss:.4f}\n",
        f"selected thresholds: {', '.join(list(map(lambda x: f'{x.threshold:.4f}', model.clust_layers)))}\n"
    ])