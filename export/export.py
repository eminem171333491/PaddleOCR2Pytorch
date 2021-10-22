import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import argparse
from pathlib import Path
from pytorchocr.base_ocr_v20 import BaseOCRV20
from collections import namedtuple
import tools.infer.pytorchocr_utility as utility


class GetNetwork(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.weights_path = args.pth_model_path
        # network_config = utility.AnalysisConfig(self.weights_path, self.yaml_path)
        weights = self.read_pytorch_weights(self.weights_path)
        self.network_config = weights['Architecture']
        weights.pop('Architecture')
        self.out_channels = self.get_out_channels(weights)
        # self.out_channels = self.get_out_channels_from_char_dict(args.rec_char_dict_path)
        kwargs['out_channels'] = self.out_channels
        super(GetNetwork, self).__init__(self.network_config, **kwargs)
        self.load_state_dict(weights)

    def __call__(self):
        return self.net


def main(args):
    model = GetNetwork(args)()

    c, h, w = args.input_shape.split(',')

    im = torch.ones(1, int(c), int(h), int(w))
    file = Path(args.pth_model_path)
    deploy_format = args.deploy_format
    if deploy_format == 'torchscript':
        export_torchscript(model, im, file)
    elif deploy_format == 'onnx':
        export_onnx(model, im, file, 11, False, False)


def export_onnx(model, im, file, opset, train, dynamic):
    import onnx

    print(f'starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')
    print(torch.__version__)
    print(torch.onnx.__path__)

    torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                      training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=not train,
                      input_names=['images'],
                      output_names=['output'],
                      dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                    'output': {0: 'batch', 2: 'out_channels'}  # shape(1,25200,85)
                                    } if dynamic else None)

    # Checks
    print("保存成功")


def export_torchscript(model, im, file):
    f = file.with_suffix('.torchscript.pt')
    trace_module = torch.jit.trace(model, im, strict=False)
    trace_module.save(f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_format", type=str, default="torchscript")
    parser.add_argument("--pth_model_path", type=str, default=r'F:\PaddleOCR2Pytorch\starneinfer.pth')
    parser.add_argument("--input_shape", type=str, default="3, 640, 640")
    opt = parser.parse_args()
    main(opt)