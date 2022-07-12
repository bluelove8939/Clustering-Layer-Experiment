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


model = torchvision.models.resnet50(pretrained=True).to(device)

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
save_modelname = "Resnet50_STL10_normal.pth"
save_fullpath = os.path.join(save_dirpath, save_modelname)

def show_activations(model, channel_size=9):
    save_image_dirpath = os.path.join(os.curdir, 'model_activations')
    if not os.path.exists(save_image_dirpath):
        os.makedirs(save_image_dirpath)
    save_imagename = f"Resnet50_STL10_normal_channel_{channel_size}.png"
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

    fig, axs = plt.subplots(cgrid, rgrid)
    fig.set_size_inches(3, 3)
    fig.suptitle("Normal Intermediate Activation Images")

    ridx, cidx = 0, 0
    for key in activation.keys():
        act = activation[key].squeeze()
        for ridx in range(rgrid):
            if ridx < act.size(0):
                axs[cidx, ridx].imshow(act[ridx].to('cpu'))
            else:
                axs[cidx, ridx].axis('off')
        cidx += 1

    plt.tight_layout()
    plt.savefig(save_image_fullpath)
    plt.show()


if __name__ == '__main__':
    epoch = 5
    for eidx in range(epoch):
        print(f"\nEpoch: {eidx}")
        train(train_loader, model, loss_fn=loss_fn, optimizer=optimizer)
    test(test_loader, model, loss_fn=loss_fn)

    if 'model_output' not in os.listdir(os.curdir):
        os.mkdir(os.path.join(os.curdir, 'model_output'))
    torch.save(model.state_dict(), save_fullpath)

    # show_activations(model, channel_size=9)
