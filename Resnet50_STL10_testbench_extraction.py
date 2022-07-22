import os
import torch

import Resnet50_STL10_normal as normal
import Resnet50_STL10_clustered as clustered
from tools.extractor import ActivationExtractor, ParamsExtractor


normal.model.load_state_dict(torch.load(normal.save_fullpath))
output_modelname = 'Resnet50_STL10_normal'
savepath = os.path.join(os.curdir, 'model_activations_raw')

normal_act_extractor = ActivationExtractor(normal.model, output_modelname=output_modelname, device=normal.device)
normal_act_extractor.register_hook(normal.model.conv1, 'conv1')
normal_act_extractor.register_hook(normal.model.layer1, 'layer1')
normal_act_extractor.register_hook(normal.model.layer2, 'layer2')
normal_act_extractor.register_hook(normal.model.layer3, 'layer3')
normal_act_extractor.register_hook(normal.model.layer4, 'layer4')
normal_act_extractor.register_hook(normal.model.fc, 'fc')
normal_act_extractor.extract_activation(dataloader=normal.test_loader, max_iter=5)
normal_act_extractor.save_activation(savepath=savepath)

normal_param_extractor = ParamsExtractor(normal.model, output_modelname=output_modelname)
normal_param_extractor.add_trace('weight')
normal_param_extractor.extract_params()
normal_param_extractor.save_params()