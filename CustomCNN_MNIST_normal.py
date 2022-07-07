import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt

from tools.progressbar import progressbar

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NetworkModel(nn.Module):
    def __init__(self):
        super (NetworkModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Sequential(
            nn.Linear(294, 500),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


train_dataset = datasets.MNIST(
    root=os.path.join(os.curdir, '../data'),
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
root=os.path.join(os.curdir, '../data'),
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

model = NetworkModel()

loss_fn = nn.CrossEntropyLoss()  # loss functions
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizer module

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
            print(f"\rtest status: {progressbar(didx, len(dataloader), scale=50)} {didx / len(dataloader) * 100:2.0f}%",end='')
    test_loss /= num_batches   # make an average of the total loss
    correct /= size            # make an average with correctness count
    print(f"\nTest Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct, test_loss

save_dirpath = os.path.join(os.curdir, 'model_output')
save_modelname = "CustomCNN_MNIST_normal.pth"
save_fullpath = os.path.join(save_dirpath, save_modelname)

if __name__ == '__main__':
    train(train_loader, model, loss_fn=loss_fn, optimizer=optimizer)
    test(test_loader, model, loss_fn=loss_fn)

    if 'model_output' not in os.listdir(os.curdir):
        os.mkdir(os.path.join(os.curdir, 'model_output'))
    torch.save(model.state_dict(), save_fullpath)


# activation = {}
#
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
#
#
# model.conv1.register_forward_hook(get_activation('conv1'))
# model.conv2.register_forward_hook(get_activation('conv2'))
# data, _ = test_dataset[0]
# data.unsqueeze_(0)
# model.eval()
# output = model(data)
#
# rgrid, cgrid = 0, 0
# for key in activation.keys():
#     rgrid = max(rgrid, activation[key].squeeze().size(0))
#     cgrid += 1
#
# fig, axs = plt.subplots(cgrid, rgrid)
# fig.suptitle("Intermediate Activation Images")
#
# ridx, cidx = 0, 0
# for key in activation.keys():
#     act = activation[key].squeeze()
#     for ridx in range(rgrid):
#         if ridx < act.size(0):
#             axs[cidx, ridx].imshow(act[ridx])
#             axs[cidx, ridx].set_title(f"{key} channel{ridx}")
#         else:
#             axs[cidx, ridx].axis('off')
#     cidx += 1
#
# plt.show()
