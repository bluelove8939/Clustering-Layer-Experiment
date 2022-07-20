import os
import copy
import torch
import torch.quantization.quantize_fx as quantize_fx
from torch.fx import Interpreter

from tools.progressbar import progressbar
from tools.training import train, test


class QuantizationModule(object):
    def __init__(self, tuning_dataloader, loss_fn, optimizer):
        self.tuning_dataloader = tuning_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def quantize(self, model, default_qconfig='fbgemm', calib=True, verbose=2, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print("\nQuantization Configs")
        print(f"- loss_fn: {self.loss_fn}")
        print(f"- qconfig: {default_qconfig}")
        print(f"- device:  {device}")

        quant_model = copy.deepcopy(model)
        quant_model.eval()
        qconfig = torch.quantization.get_default_qconfig(default_qconfig)
        qconfig_dict = {"": qconfig}

        if verbose: print("preparing quantization (symbolic tracing)")
        model_prepared = quantize_fx.prepare_fx(quant_model, qconfig_dict)
        if verbose == 2: print(model_prepared)

        if calib: self.calibrate(model_prepared, verbose=verbose, device=device)
        model_quantized = quantize_fx.convert_fx(model_prepared)
        return model_quantized

    def calibrate(self, model, verbose=2, device="auto"):
        if verbose == 1:
            print(f'\r{progressbar(0, len(self.tuning_dataloader), 50)}'
                  f'  calibration iter: {0:3d}/{len(self.tuning_dataloader):3d}', end='')
        elif verbose:
            print(f'calibration iter: {0:3d}/{len(self.tuning_dataloader):3d}')

        cnt = 1
        model.eval()                                      # set to evaluation mode

        with torch.no_grad():                             # do not save gradient when evaluation mode
            for image, target in self.tuning_dataloader:  # extract input and output data
                image = image.to(device)
                model(image)                              # forward propagation

                if verbose == 1:
                    print(f'\r{progressbar(cnt, len(self.tuning_dataloader), 50)}'
                          f'  calibration iter: {cnt:3d}/{len(self.tuning_dataloader):3d}', end='')
                elif verbose:
                    print(f'calibration iter: {cnt:3d}/{len(self.tuning_dataloader):3d}', end='')

                cnt += 1

        if verbose == 1: print('\n')
        elif verbose: print()


class QuantizedModelExtractor(Interpreter):
    def __init__(self, gm, output_modelname='model', device='auto'):
        super(QuantizedModelExtractor, self).__init__(gm)
        self.output_modelname = output_modelname
        self.features = {}
        self.traces = []
        self.device = device
        self.target_model = gm

        if device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def call_module(self, target, *args, **kwargs):
        for kw in self.traces:
            if kw in target.split('.'):
                idx = 0
                save_output_name = f"{self.output_modelname}_{target}_output{idx}"
                if save_output_name in self.features:
                    idx += 1
                    save_output_name = f"{self.output_modelname}_{target}_output{idx}"

                print(f'extracting {save_output_name}')
                self.features[save_output_name] = super().call_module(target, *args, **kwargs)
        return super().call_module(target, *args, **kwargs)

    def add_trace(self, name):
        self.traces.append(name)

    def save_features(self, savepath=None):
        if savepath is None:
            savepath = os.path.join(os.curdir, 'model_activations_raw', self.output_modelname)
        os.makedirs(savepath, exist_ok=True)

        for layer_name in self.features.keys():
            torch.save(self.features[layer_name], os.path.join(savepath, f"{layer_name}"))

        with open(os.path.join(self.savepath, 'filelist.txt'), 'wt') as filelist:
            filelist.write('\n'.join([os.path.join(savepath, layer_name) for layer_name in self.features.keys()]))

    def extract_activations(self, dataloader, max_iter=5):
        iter_cnt = 0

        for X, y in dataloader:
            if iter_cnt > max_iter: break
            else: iter_cnt += 1
            X, y = X.to(self.device), y.to(self.device)
            self.run(X)

    def extract_parameters(self):
        for param_name in self.target_model.state_dict():
            if 'weight' in param_name:
                parsed_name = f"{self.output_modelname}_{param_name.replace('.', '_')}"
                try:
                    print(f"extracting {parsed_name}")
                    self.features[parsed_name] = self.target_model.state_dict()[param_name].int_repr().detach()
                except:
                    print(f"error occurred on extracting {parsed_name}")