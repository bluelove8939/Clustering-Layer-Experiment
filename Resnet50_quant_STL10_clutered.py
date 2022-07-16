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

from tools.progressbar import progressbar
from custom_layers import clustering

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def get_mean(dataset):
  meanRGB = [np.mean(image.numpy(), axis=(1,2)) for image,_ in dataset]
  meanR = np.mean([m[0] for m in meanRGB])
  meanG = np.mean([m[1] for m in meanRGB])
  meanB = np.mean([m[2] for m in meanRGB])
  return [meanR, meanG, meanB]

def get_std(dataset):
  stdRGB = [np.std(image.numpy(), axis=(1,2)) for image,_ in dataset]
  stdR = np.mean([s[0] for s in stdRGB])
  stdG = np.mean([s[1] for s in stdRGB])
  stdB = np.mean([s[2] for s in stdRGB])
  return [stdR, stdG, stdB]

train_dataset = datasets.STL10(
    root=os.path.join(os.curdir, '../data'),
    split='train',
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.STL10(
    root=os.path.join(os.curdir, '../data'),
    split='test',
    download=True,
    transform=transforms.ToTensor()
)

train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(get_mean(train_dataset), get_std(train_dataset))])
test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(get_mean(test_dataset), get_std(test_dataset))])

train_dataset.transforms = train_transforms
test_dataset.transforms = test_transforms

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


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


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # size of the dataset
    model.train()                   # turn the model into train mode
    for batch, (X, y) in enumerate(dataloader):  # each index of dataloader will be batch index
        X, y = X.to(device), y.to(device)        # extract input and output

        # Compute prediction error
        pred = model(X)          # predict model
        # print(pred.shape, y.shape)
        loss = loss_fn(pred, y)  # calculate loss

        # Backpropagation
        optimizer.zero_grad()  # gradient initialization (just because torch accumulates gradient)
        loss.backward()        # backward propagate with the loss value (or vector)
        optimizer.step()       # update parameters

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # dataset size
    num_batches = len(dataloader)   # the number of batches
    model.eval()                    # convert model into evaluation mode
    test_loss, correct = 0, 0       # check total loss and count correctness
    with torch.no_grad():           # set all of the gradient into zero
        for didx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)     # extract input and output
            pred = model(X)                       # predict with the given model
            test_loss += loss_fn(pred, y).item()  # acculmulate total loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # count correctness
            print(f"\rtest status: {progressbar(didx+1, len(dataloader), scale=50)} {(didx+1) / len(dataloader) * 100:2.0f}%",end='')
    test_loss /= num_batches   # make an average of the total loss
    correct /= size            # make an average with correctness count
    print(f"\nTest Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct, test_loss


save_dirpath = os.path.join(os.curdir, 'model_output')
if not os.path.exists(save_dirpath):
    os.makedirs(save_dirpath)
save_modelname = "Resnet50_quant_STL10_clustered.pth"
save_fullpath = os.path.join(save_dirpath, save_modelname)


if __name__ == '__main__':
    from tools.quanitzation import QuantizationModule
    import Resnet50_STL10_normal as normal

    total_datasize = len(train_dataset)
    valid_datasize = int(total_datasize / 10)
    tuning_dataset, _ = torch.utils.data.random_split(train_dataset, [valid_datasize, total_datasize - valid_datasize])
    tuning_loader = DataLoader(tuning_dataset, batch_size=10)

    quant_module = QuantizationModule(tuning_loader, loss_fn=loss_fn, optimizer=optimizer)
    model.load_state_dict(torch.load(normal.save_fullpath))
    quanitzed_model = quant_module.quantize(model, default_qconfig='fbgemm', verbose=1)

    torch.save(quanitzed_model.state_dict(), save_fullpath)