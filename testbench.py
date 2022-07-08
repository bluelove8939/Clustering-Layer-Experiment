import os
import torch
from torch.utils.data import DataLoader

import CustomCNN_MNIST_normal as normal
import CustomCNN_MNIST_clustered as clustered

resultfile_path = os.path.join(os.curdir, 'logs', 'clustering_test_results.csv')
thresholds = [10.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
# thresholds = [0.000000000000000000001]
clustering_test_results = ["Thres,Acc,AvgLoss,ClustAmt1,ClustAmt2"]

total_datasize = len(clustered.test_dataset)
valid_datasize = int(total_datasize * 0.01)
test_dataset, _ = torch.utils.data.random_split(clustered.test_dataset, [valid_datasize, total_datasize-valid_datasize])
test_loader = DataLoader(test_dataset, batch_size=5)
loss_fn = clustered.loss_fn

model = normal.NetworkModel()
model.load_state_dict(torch.load(normal.save_fullpath))

print('testing with normal model')
accuracy, avg_loss = clustered.test(test_loader, model, loss_fn)

clustering_test_results.append(','.join(map(str, [
    0, accuracy, avg_loss, 0, 0,
])))

for thres in thresholds:
    clustered.clustering.set_logger(info=f"thres_{thres}", verbose=False)

    model = clustered.NetworkModel()
    model.load_state_dict(torch.load(normal.save_fullpath))
    model.reset_clust_layer()
    model.set_clust_threshold(thres)

    print(f'testing with threshold value: {thres}')
    accuracy, avg_loss = clustered.test(test_loader, model, loss_fn)

    clustering_test_results.append(','.join(map(str, [
        thres, accuracy, avg_loss, model.clust1.get_clust_amt(), model.clust2.get_clust_amt(),
    ])))

    with open(resultfile_path, 'wt') as file:
        file.write('\n'.join(clustering_test_results))