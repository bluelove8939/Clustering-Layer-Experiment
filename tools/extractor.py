import os
import torch


class ActivationExtractor(object):
    def __init__(self, target_model, output_modelname='model', device='auto'):
        self.target_model = target_model
        self.output_modelname = output_modelname
        self._hook_names = []
        self._activation = {}
        self.device = device

        if self.device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def register_hook(self, layer: torch.nn.Module, name='auto'):
        if name == 'auto':
            hidx = 0
            while f"hook{hidx}" in self._hook_names: hidx += 1
            name = f"hook{hidx}"
        layer.register_forward_hook(self._extract_hook(name))

    def _extract_hook(self, name):
        def hook(model, layer_input, layer_output):
            oidx = 0
            while f"{name}_output{oidx}" in self._activation.keys(): oidx += 1
            save_output_name = f"{name}_output{oidx}"
            self._activation[save_output_name] = layer_output.detach()
            print(f'extracting {save_output_name} (type: {self._activation[save_output_name].type()})')

        return hook

    def extract_activation(self, dataloader, max_iter=5):
        iter_cnt = 0

        for X, y in dataloader:
            if iter_cnt > max_iter:
                break
            else:
                iter_cnt += 1
            X, y = X.to(self.device), y.to(self.device)
            self.target_model(X)

    def save_activation(self, savepath=None):
        if savepath is None:
            savepath = os.path.join(os.curdir, 'model_activations_raw', self.output_modelname)
        os.makedirs(savepath, exist_ok=True)

        for layer_name in self._activation.keys():
            barr = self._activation[layer_name].detach().numpy()
            with open(os.path.join(savepath, f"{layer_name}"), 'wb') as file:
                file.write(barr)

        with open(os.path.join(savepath, 'filelist.txt'), 'wt') as filelist:
            filelist.write('\n'.join([os.path.join(savepath, layer_name) for layer_name in self._activation.keys()]))


class ParamsExtractor(object):
    def __init__(self, target_model, output_modelname='model'):
        self.target_model = target_model
        self.output_modelname = output_modelname
        self._params = {}
        self.traces = []

    def add_trace(self, trace):
        if trace not in self.traces:
            self.traces.append(trace)

    def extract_params(self):
        for param_name in self.target_model.state_dict():
            for trace in self.traces:
                if trace in param_name:
                    parsed_name = f"{self.output_modelname}_{param_name.replace('.', '_')}"
                    try:
                        print(f"extracting {parsed_name}")
                        self._params[parsed_name] = self.target_model.state_dict()[param_name].detach()
                    except:
                        print(f"error occurred on extracting {parsed_name}")
                    break

    def save_params(self, savepath=None):
        if savepath is None:
            savepath = os.path.join(os.curdir, 'model_activations_raw', self.output_modelname)
        os.makedirs(savepath, exist_ok=True)

        for layer_name in self._params.keys():
            barr = self._params[layer_name].detach().numpy()
            with open(os.path.join(savepath, f"{layer_name}"), 'wb') as file:
                file.write(barr)