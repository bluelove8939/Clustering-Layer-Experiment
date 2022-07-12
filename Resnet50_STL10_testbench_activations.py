import os
from itertools import product

import torch
from torch.utils.data import DataLoader

import Resnet50_STL10_normal as normal
import Resnet50_STL10_clustered as clustered


# Test dataset generation
total_datasize = len(clustered.test_dataset)
valid_datasize = int(total_datasize / 40)
print(f"test dataset size: {valid_datasize}/{total_datasize}")
test_dataset, _ = torch.utils.data.random_split(clustered.test_dataset, [valid_datasize, total_datasize-valid_datasize])
test_loader = DataLoader(test_dataset, batch_size=10)
loss_fn = clustered.loss_fn

normal_model = normal.model
normal_model.load_state_dict(torch.load(normal.save_fullpath))
normal_acc, normal_avg_loss = normal.test(test_loader, normal_model, loss_fn)

model = clustered.model
model.load_state_dict(torch.load(normal.save_fullpath))
model.reset_clust_layer()
model.set_clust_threshold(*[0.1, 0.1, 0.1, 0.01, 0.01])
clust_acc, clust_avg_loss = clustered.test(test_loader, model, loss_fn)

print(f"normal test result:    acc({normal_acc}), avg loss({normal_avg_loss:.6f})")
print(f"clustered test result: acc({clust_acc}), avg loss({clust_avg_loss:.6f})")
normal.show_activations(normal_model, channel_size=9)
clustered.show_activations(model, channel_size=9)