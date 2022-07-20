import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import matplotlib.pyplot as plt

from tools.imagenet_utils.args_generator import args
from tools.imagenet_utils.training import train, validate
from tools.progressbar import progressbar
from custom_layers import clustering

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Dataset configuration
dataset_dirname = args.data

train_dataset = datasets.ImageFolder(
        os.path.join(dataset_dirname, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

test_dataset = datasets.ImageFolder(
        os.path.join(dataset_dirname, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# Customize pretrained reference model
class NetworkModel(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=10):
        super(NetworkModel, self).__init__(block, layers, num_classes=num_classes)
        self.clust1 = clustering.QuantClusteringLayer(threshold=0.05, cacheline_size=64)
        self.clust2 = clustering.QuantClusteringLayer(threshold=0.05, cacheline_size=64)
        self.clust3 = clustering.QuantClusteringLayer(threshold=0.05, cacheline_size=64)
        self.clust4 = clustering.QuantClusteringLayer(threshold=0.05, cacheline_size=64)
        self.clust5 = clustering.QuantClusteringLayer(threshold=0.05, cacheline_size=64)
        self.clust_layers = [self.clust1, self.clust2, self.clust3, self.clust4, self.clust5]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.clust1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.clust2(x)
        x = self.layer2(x)
        x = self.clust3(x)
        x = self.layer3(x)
        x = self.clust4(x)
        x = self.layer4(x)
        x = self.clust5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_clust_threshold(self, *args):
        for lay, thres in zip(self.clust_layers, args):
            lay.threshold = thres

    def get_clust_amt(self):
        return list(map(lambda x: x.get_clust_amt(), self.clust_layers))

    def get_clust_base_cnt(self):
        return list(map(lambda x: x.get_clust_base_cnt(), self.clust_layers))

    def reset_clust_layer(self):
        for lay in self.clust_layers:
            lay.reset_clust_amt()

model = NetworkModel(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 1000).to(device)
print(model)

lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss().to(device)


save_dirpath = os.path.join(os.curdir, 'model_output')
if not os.path.exists(save_dirpath):
    os.makedirs(save_dirpath)
save_modelname = "Resnet50_quant_Imagenet_clustered.pth"
save_fullpath = os.path.join(save_dirpath, save_modelname)


if __name__ == '__main__':
    from tools.quanitzation import QuantizationModule
    import Resnet50_Imagenet_normal as normal

    total_datasize = len(train_dataset)
    valid_datasize = int(total_datasize / 10)
    tuning_dataset, _ = torch.utils.data.random_split(train_dataset, [valid_datasize, total_datasize - valid_datasize])
    tuning_loader = DataLoader(tuning_dataset, batch_size=10)

    quant_module = QuantizationModule(tuning_loader, loss_fn=loss_fn, optimizer=optimizer)
    model.load_state_dict(torch.load(normal.save_fullpath))
    model.set_clust_threshold(0.01, 0.05, 0.02, 0.02, 0.1)
    quantized_model = quant_module.quantize(model, default_qconfig='fbgemm', verbose=1)

    torch.save(quantized_model.state_dict(), save_fullpath)

    validate(test_loader, quantized_model, loss_fn, args)