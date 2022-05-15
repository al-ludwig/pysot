import torch
import argparse
import os

from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg

# argparse check function
def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"Config file:{path} is not a valid file")

# argument parsing
parser = argparse.ArgumentParser(description='Script for adding a fc layer to a model.')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('--snapshot', required=True, type=file_path, help="Path to the model file (.pth)")
required.add_argument('--cfg', required=True, type=file_path)
args = parser.parse_args()

class BackboneBuilder(ModelBuilder):
    def __init__(self):
        super(BackboneBuilder, self).__init__()
    
    # override:
    def forward(self, data):
        return self.backbone(data)

def main():
    # load cfg
    cfg.merge_from_file(args.cfg)
    snapshot = torch.load(args.snapshot,  map_location=torch.device('cpu'))

    model = BackboneBuilder()
    for key in list(snapshot.keys()):
        if not key.startswith('backbone'):
            del snapshot[key]
    model.load_state_dict(snapshot, strict=False)

    x = torch.randn(1, 3, 255, 255, requires_grad=True)
    # torch_out = model(x)

    outputfile = os.path.abspath(args.snapshot).split('.')
    outputfile[0] = outputfile[0] + '_withfc'
    outputfile = outputfile[0] + "." + outputfile[1]

    torch.save({'state_dict': model.backbone.state_dict()}, outputfile)
    return

if __name__ == '__main__':
    main()