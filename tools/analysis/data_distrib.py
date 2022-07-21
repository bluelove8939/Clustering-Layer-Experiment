import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np


model_name = "Resnet50_p0.7_quantized"
arrsize = 5000
output_idx = 1
target_dirname = os.path.join('..', '..', 'model_activations_raw', model_name)

layers = list()
layers.append("conv1")
for lidx, slim in zip([1, 2, 3, 4], [3, 4, 6, 3]):
    for sidx in range(slim):
        for cidx in range(3):
            layers.append(f"layer{lidx}.{sidx}.conv{cidx+1}")
layers.append("fc")

rgrid = 7
cgrid = math.ceil(len(layers) / rgrid)
fig, axs = plt.subplots(cgrid, rgrid, figsize=(3 * rgrid, 3 * cgrid),
                        gridspec_kw={'width_ratios': [1] * rgrid, 'height_ratios': [1] * cgrid},
                        constrained_layout=True)
fig.suptitle(model_name)

ridx, cidx = 0, 0
for layer_name in layers:
    target_filepath = os.path.join(target_dirname, f"{model_name}_{layer_name}_output{output_idx}")

    with open(target_filepath, 'rb') as file:
        content = np.frombuffer(file.read(arrsize), dtype=np.int8)
        q1 = np.percentile(content, 25)
        q3 = np.percentile(content, 75)
        print(f"{layer_name:15s}: q1({q1:2.0f}) q3({q3:2.0f})")

    axs[cidx, ridx].hist(content, bins=200)
    axs[cidx, ridx].set_title(f"{layer_name}")

    ridx += 1
    if ridx == rgrid:
        cidx += 1
        ridx = 0

for aidx in range(len(layers), rgrid * cgrid):
    axs[math.floor(aidx / rgrid), aidx % rgrid].axis('off')

# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()