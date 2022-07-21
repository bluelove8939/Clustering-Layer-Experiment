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

# from tools.progressbar import progressbar
from tools.training import train, test
from tools.pruning import PruneModule
from tools.quanitzation import QuantizationModule, QuantizedModelExtractor

# parse commandline args
import argparse
parser = argparse.ArgumentParser(description='Resnet50 training config')
parser.add_argument('--skip-training', default=False, action='store_true',
                    help='skips training (bool)')
args = parser.parse_args()


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
    split='train',
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

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

# Model Configs
model_type = "Resnet50"
model = torchvision.models.resnet50(pretrained=True).to(device)

# Training Configs
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss().to(device)

# Quantization Configs
qmodule = QuantizationModule(train_loader, loss_fn=loss_fn, optimizer=optimizer)

# Model Saving Configs
save_dirpath = os.path.join(os.curdir, 'model_output')
save_fullpath_normal = os.path.join(save_dirpath, "Resnet50_STL10_normal.pth")
if not os.path.exists(save_dirpath):
    os.makedirs(save_dirpath)

def save_fullpath(quantized=False):
    save_modelname = f"Resnet50{'_quantized' if quantized else ''}_STL10_normal.pth"
    save_fullpath = os.path.join(save_dirpath, save_modelname)
    return save_fullpath


def show_activations(model, channel_size=9):
    save_image_dirpath = os.path.join(os.curdir, 'model_activations')
    if not os.path.exists(save_image_dirpath):
        os.makedirs(save_image_dirpath)
    save_imagename = f"{save_fullpath(quantized=True).split('.')[0]}_csize{channel_size}.png"
    save_image_fullpath = os.path.join(save_image_dirpath, save_imagename)

    model.load_state_dict(torch.load(save_fullpath(quantized=True)))
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
    if not args.skip_training:
        for eidx in range(epoch):
            print(f"\nEpoch: {eidx}")
            train(train_loader, model, loss_fn=loss_fn, optimizer=optimizer, verbose=1)
        test(test_loader, model, loss_fn=loss_fn, verbose=1)
    else:
        print("skip training and load saved state dict")
        print(f"state dict: {save_fullpath_normal}")
        model.load_state_dict(torch.load(save_fullpath_normal))

    # Quantizing model
    qmodel = qmodule.quantize(model, default_qconfig='fbgemm', calib=True, verbose=1)
    test(test_loader, qmodel, loss_fn=loss_fn, verbose=1, tdevice='cpu')

    # Extract output activations
    qextractor = QuantizedModelExtractor(qmodel, output_modelname=f"{model_type}_quantized", device='cpu')
    qextractor.add_trace('conv')
    qextractor.add_trace('conv1')
    qextractor.add_trace('conv2')
    qextractor.add_trace('conv3')
    qextractor.add_trace('conv4')
    qextractor.add_trace('conv5')
    qextractor.add_trace('fc')
    qextractor.extract_activations(test_loader, max_iter=5)
    qextractor.extract_parameters()
    qextractor.save_features(savepath=None)

    if 'model_output' not in os.listdir(os.curdir):
        os.mkdir(os.path.join(os.curdir, 'model_output'))
    torch.save(qmodel.state_dict(), save_fullpath(quantized=True))

    # show_activations(model, channel_size=9)
