import torch
from torch import nn
from .output import Output
from .merger import Merger
from .vgg import vgg16


class East(nn.Module):
    def __init__(self, in_channels ,pretrained = True, **kwargs):
        super(East, self).__init__()

        self.feature_extractor = vgg16(in_channels, pretrained, **kwargs)
        self.merger = Merger()
        self.output = Output()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.merger(x)
        x = self.output(x)
        return x


if __name__ == "__main__":

    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_input = torch.randn(1, 3, 1280, 640).to(device)
    east = East(3, batch_norm = True).to(device)
    east.eval()
    with torch.no_grad():
        since = time.time()
        out = east(test_input)

    print(f"Total time: {time.time() - since}")