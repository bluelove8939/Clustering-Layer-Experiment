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
from tools.imagenet_utils.args_generator import args
from tools.imagenet_utils.training import train, validate


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Dataset configuration
dataset_dirname = os.path.join(os.path.abspath(os.sep), 'shared', 'Imagenet_data')

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


# Model configuration
model = torchvision.models.resnet50(pretrained=True).to(device)

lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss().to(device)


save_dirpath = os.path.join(os.curdir, 'model_output')
if not os.path.exists(save_dirpath):
    os.makedirs(save_dirpath)
save_modelname = "Resnet50_Imagenet_normal.pth"
save_fullpath = os.path.join(save_dirpath, save_modelname)

def show_activations(model, channel_size=9):
    save_image_dirpath = os.path.join(os.curdir, 'model_activations')
    if not os.path.exists(save_image_dirpath):
        os.makedirs(save_image_dirpath)
    save_imagename = f"Resnet50_Imagenet_normal_channel_{channel_size}.png"
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
    epoch = 5
    # train(train_loader, model, loss_fn, optimizer, epoch, args)
    validate(test_loader, model, loss_fn, args)

    if 'model_output' not in os.listdir(os.curdir):
        os.mkdir(os.path.join(os.curdir, 'model_output'))
    torch.save(model.state_dict(), save_fullpath)

    # show_activations(model, channel_size=9)
