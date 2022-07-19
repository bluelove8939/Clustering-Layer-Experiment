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
parser.add_argument('--save-log', default=False, action='store_true',
                    help='saves log (bool)')
parser.add_argument('-e', '--epoch', default=100, type=int,
                    help='number of epoch (int)')
parser.add_argument('-lr', '--learning-rate', dest='lr', default=0.1, type=float,
                    help='learning rate (float)')
parser.add_argument('-m', '--momentum', dest='momentum', default=0.9, type=float,
                    help='learning rate (float)')
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

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
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
    if 'model_output' not in os.listdir(os.curdir):
        os.mkdir(os.path.join(os.curdir, 'model_output'))
    if args.resume or args.skip_training:
        model.load_state_dict(torch.load(save_fullpath))

    epoch = args.epoch
    logs_path = os.path.join(os.curdir, 'logs', 'Resnet50_CIFAR10_normal.log')

    if args.save_log:
        with open(logs_path, 'wt') as logfile:
            logfile.write('test config\n')
            if not args.skip_training:
                logfile.write(f'- epoch: {epoch}\n')
                logfile.write(f'- resume: {args.resume}\n')
            else:
                logfile.write('- skip training: True')

    print('\ntest config')
    print(f'- save path: {save_fullpath}')
    print(f'- logs path: {logs_path}')
    if not args.skip_training:
        print(f'- epoch: {epoch}')
        print(f'- resume: {args.resume}')

        for eidx in range(epoch):
            print(f"\nEpoch: {eidx}")
            if args.save_log:
                with open(logs_path, 'at') as logfile:
                    logfile.write(f"\nEpoch: {eidx}\n")
            scheduler.step()
            torch.save(model.state_dict(), save_fullpath)
    else:
        print('- skip training: True')

    print('\nValidation test with trainset')
    test(train_loader, model, loss_fn=loss_fn, verbose=1, savelog_path=logs_path)
    print('\nValidation test with testset')
    test(test_loader, model, loss_fn=loss_fn, verbose=1, savelog_path=logs_path)

    # show_activations(model, channel_size=9)
