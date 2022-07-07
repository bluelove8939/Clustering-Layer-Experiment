import os
import torch

import CustomCNN_MNIST_normal as normal
import CustomCNN_MNIST_clustered as clustered

resultfile_path = os.path.join(os.curdir, 'logs', 'clustering_test_results.csv')
thresholds = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
clustering_test_results = ["Thres,Acc,AvgLoss,ClustAmt1,ClustAmt2"]

for thres in thresholds:
    model = clustered.NetworkModel()
    model.load_state_dict(torch.load(normal.save_fullpath))
    model.reset_clust_layer()
    model.set_clust_threshold(thres)

    test_loader = clustered.test_loader
    loss_fn = clustered.loss_fn

    accuracy, avg_loss = clustered.test(test_loader, model, loss_fn)

    clustering_test_results.append(','.join([
        thres, accuracy, avg_loss, model.clust1.get_clust_amt(), model.clust2.get_clust_amt(),
    ]))

with open(resultfile_path, 'wt') as file:
    file.write('\n'.join(clustering_test_results))