import torch
from torch import nn
import traceback
from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

#VGG16 block
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def remove_classifier_keys(state_dict):

    new_state_dict = state_dict.copy()
    for key in state_dict.keys():
        if key.find("classifier") != -1:
            try:
                new_state_dict.pop(key)
            except:
                traceback.print_exc()

    return new_state_dict


class VGG(nn.Module):
    def __init__(self, in_channels, cfg, batch_norm = True, init_weights=True):
        super(VGG, self).__init__()
        self.features = self.make_layers(in_channels, cfg, batch_norm)

        if init_weights:
            self._initialize_weights()
    

    def make_layers(self, in_channels, cfg, batch_norm):

        layers = []

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)


    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)

        return out[1:] #feature map from pooling2 to pooling5 extracted 



def vgg16(in_channels, pretrained, **kwargs):
    '''
    Function to load vgg16 features model
    '''
    if pretrained:
        kwargs['init_weights'] = False
    
    model = VGG(in_channels, cfg, **kwargs)

    #download model from url
    if pretrained:
        if kwargs['batch_norm']:
            url = model_urls['vgg16_bn']
        else:
            url = model_urls['vgg16']

        state_dict = load_state_dict_from_url(url, progress = True)
        state_dict = remove_classifier_keys(state_dict)
        model.load_state_dict(state_dict)

    return model