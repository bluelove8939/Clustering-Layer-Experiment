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

from tools.training import test, train
from tools.progressbar import progressbar

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', default=False, action='store_true',
                    help='resume with the stored state dict (bool)')
parser.add_argument('--skip-training', default=False, action='store_true',
                    help='skips training (bool)')
parser.add_argument('--epoch', default=100, type=int,
                    help='number of epoch (int)')
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Preparing Datasets
dataset_name = 'CIFAR10'
transformer = transforms.Compose([transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transformer)
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transformer)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=200, shuffle=False)


# Model configs
model_name = 'Resnet50'
model_type = 'normal'
model = torchvision.models.resnet50().to(device)

lr = 0.1
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
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
    if args.resume or args.skip_training:
        model.load_state_dict(torch.load(save_fullpath))
    epoch = args.epoch

    print('\ntest config')
    print(f'- save path: {save_fullpath}')
    if not args.skip_training:
        print(f'- epoch: {epoch}')
        print(f'- resume: {args.resume}')

        for eidx in range(epoch):
            print(f"\nEpoch: {eidx}")
            train(train_loader, model, loss_fn=loss_fn, optimizer=optimizer, verbose=1)
            # scheduler.step()
            torch.save(model.state_dict(), save_fullpath)
        test(test_loader, model, loss_fn=loss_fn, verbose=1)
    else:
        print('- skip training: True')

    if 'model_output' not in os.listdir(os.curdir):
        os.mkdir(os.path.join(os.curdir, 'model_output'))
    torch.save(model.state_dict(), save_fullpath)

    # show_activations(model, channel_size=9)
