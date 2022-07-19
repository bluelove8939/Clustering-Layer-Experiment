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

from custom_layers import clustering
from tools.training import test, train
from tools.progressbar import progressbar

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Customize pretrained reference model
class NetworkModel(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=10):
        super(NetworkModel, self).__init__(block, layers, num_classes=num_classes)
        self.clust1 = clustering.ClusteringLayer(threshold=0.05, cacheline_size=64)
        self.clust2 = clustering.ClusteringLayer(threshold=0.05, cacheline_size=64)
        self.clust3 = clustering.ClusteringLayer(threshold=0.05, cacheline_size=64)
        self.clust4 = clustering.ClusteringLayer(threshold=0.05, cacheline_size=64)
        self.clust5 = clustering.ClusteringLayer(threshold=0.05, cacheline_size=64)
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


# Preparing Datasets
dataset_name = 'CIFAR10'
transformer = transforms.Compose([transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transformer)
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transformer)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)


# Model configs
model_name = 'Resnet50'
model_type = 'clustered'
model = NetworkModel(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 1000).to(device)

lr = 0.1
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss().to(device)

save_dirpath = os.path.join(os.curdir, 'model_output')
if not os.path.exists(save_dirpath):
    os.makedirs(save_dirpath)
save_modelname = f"{model_name}_{dataset_name}_{model_type}.pth"
save_fullpath = os.path.join(save_dirpath, save_modelname)

def show_activations(model, channel_size=9):
    save_image_dirpath = os.path.join(os.curdir, 'model_activations')
    if not os.path.exists(save_image_dirpath):
        os.makedirs(save_image_dirpath)
    save_imagename = f"{model_name}_{dataset_name}_{model_type}_channel_{channel_size}.png"
    save_image_fullpath = os.path.join(save_image_dirpath, save_imagename)

    model.load_state_dict(torch.load(save_fullpath))
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()[0][:channel_size]
            print(f"{name} called -> output shape: {output.detach().shape}")

        return hook

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4.register_forward_hook(get_activation('layer4'))
    data, _ = test_dataset[0]
    data.unsqueeze_(0)
    model.eval()
    output = model(data.to(device))

    rgrid, cgrid = 0, 0
    for key in activation.keys():
        rgrid = max(rgrid, activation[key].squeeze().size(0))
        cgrid += 1
    print(f"rgrid: {rgrid}, cgrid: {cgrid}")

    fig, axs = plt.subplots(cgrid, rgrid, figsize=(4 * rgrid, 4 * cgrid), gridspec_kw={'width_ratios': [1] * rgrid})
    fig.suptitle("Normal Intermediate Activation Images")

    ridx, cidx = 0, 0
    for key in activation.keys():
        act = activation[key].squeeze()
        for ridx in range(rgrid):
            if ridx < act.size(0):
                axs[cidx, ridx].imshow(act[ridx].to('cpu'))
                axs[cidx, ridx].set_title(f"{key}_channel{ridx}")
            else:
                axs[cidx, ridx].axis('off')
        cidx += 1

    # plt.tight_layout()
    plt.show()
    plt.savefig(save_image_fullpath)


if __name__ == '__main__':
    import Resnet50_CIFAR10_normal as normal
    model.load_state_dict(torch.load(normal.save_fullpath))
    model.set_clust_threshold(0.01, 0.05, 0.02, 0.02, 0.1)
    model.reset_clust_layer()
    epoch = 5
    # for eidx in range(epoch):
    #     print(f"\nEpoch: {eidx}")
    #     train(train_loader, model, loss_fn=loss_fn, optimizer=optimizer)
    test(test_loader, model, loss_fn=loss_fn, verbose=1)

    if 'model_output' not in os.listdir(os.curdir):
        os.mkdir(os.path.join(os.curdir, 'model_output'))
    torch.save(model.state_dict(), save_fullpath)

    # show_activations(model, channel_size=9)
