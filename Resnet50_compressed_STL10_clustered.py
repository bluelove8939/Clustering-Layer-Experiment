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
from custom_layers import clustering

# parse commandline args
import argparse
parser = argparse.ArgumentParser(description='Resnet50 training config')
parser.add_argument('--pamount', action='store' , type=float, default=0.3, help="pruning amount")
parser.add_argument('--step', action='store' , type=float, default=0.1, help="pruning step")
parser.add_argument('--thres', action='store' , type=float, default=1,
                    help="accuracy threshold value (difference between normal accyracy and pruned model accyracy, "
                         "percent scale)")
parser.add_argument('--tuneiter', action='store' , type=int, default=5, help="fine tuning maximum iteration value")
parser.add_argument('--skip-training', default=False, action='store_true',
                    help='skips training (bool)')
parser.add_argument('--skip-pruning', default=False, action='store_true',
                    help='skips pruning (bool)')
parser.add_argument('--skip-calib', default=False, action='store_true',
                    help='skips quantization calibration (bool)')
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


model_type = "Resnet50"
model = NetworkModel(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 1000).to(device)

# Training Configs
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss().to(device)

# Pruning Configs
prune_amount = args.pamount
pmodule = PruneModule(train_loader, loss_fn=loss_fn, optimizer=optimizer)

# Quantization Configs
qmodule = QuantizationModule(train_loader, loss_fn=loss_fn, optimizer=optimizer)

# Model Saving Configs
save_dirpath = os.path.join(os.curdir, 'model_output')
save_fullpath_normal = os.path.join(save_dirpath, "Resnet50_STL10_normal.pth")
if not os.path.exists(save_dirpath):
    os.makedirs(save_dirpath)

def save_fullpath(pamount, quantized=False):
    save_modelname = f"Resnet50_pruned{pamount * 100:.0f}{'_quantized' if quantized else ''}_STL10_normal.pth"
    save_fullpath = os.path.join(save_dirpath, save_modelname)
    return save_fullpath


def show_activations(model, channel_size=9):
    save_image_dirpath = os.path.join(os.curdir, 'model_activations')
    if not os.path.exists(save_image_dirpath):
        os.makedirs(save_image_dirpath)
    save_imagename = f"{save_fullpath(pamount=prune_amount, quantized=True).split('.')[0]}_csize{channel_size}.png"
    save_image_fullpath = os.path.join(save_image_dirpath, save_imagename)

    model.load_state_dict(torch.load(save_fullpath(pamount=prune_amount, quantized=True)))
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
    model.set_clust_threshold(0.01, 0.05, 0.02, 0.02, 0.1)

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

    # Pruning model
    if not args.skip_pruning:
        threshold = args.thres
        step = args.step
        max_iter = args.tuneiter

        pmodule.prune_model(model, target_amount=prune_amount, threshold=threshold, step=step, max_iter=max_iter,
                            pass_normal=False, verbose=1)
        pmodule.remove_prune_model(model)
        test(test_loader, model, loss_fn=loss_fn, verbose=1)
        torch.save(model.state_dict(), save_fullpath(pamount=prune_amount))
    else:
        print("skip training and load saved state dict")
        print(f"state dict: {save_fullpath(pamount=prune_amount)}")
        model.load_state_dict(torch.load(save_fullpath(pamount=prune_amount)))

    # Quantizing model
    qmodel = qmodule.quantize(model, default_qconfig='fbgemm', calib=(not args.skip_calib), verbose=1)
    if args.skip_calib:
        print('skip calibration and directly load save statedict')
        print(f"state dict: {save_fullpath(pamount=prune_amount, quantized=True)}")
        qmodel.load_state_dict(torch.load(save_fullpath(pamount=prune_amount, quantized=True)))
    qextractor = QuantizedModelExtractor(qmodel, output_modelname=f"{model_type}_p{prune_amount}_quantized_clustered",
                                         device='cpu')
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

    # if 'model_output' not in os.listdir(os.curdir):
    #     os.mkdir(os.path.join(os.curdir, 'model_output'))
    # torch.save(qmodel.state_dict(), save_fullpath(pamount=prune_amount, quantized=True))

    # show_activations(model, channel_size=9)
