import torch

from fcn_resnet import resnet50
from model_size import show_model


def getBackBone(name):
    model_map = {
        'fcnResNet50': fcnResNet50,
    }

    return model_map[name]()


def fcnResNet50():
    '''
    Transform resNet50 to fcnResNet50
     specific
     The original resNet output image category,
        the original image is reduced by 32 times after the last big layer,
        and enter avgpool, fc, output image category
     1. Remove the last avgpool, fc layer
     2. Let the model output the output of each layer,
        which is reduced by 4 times, reduced by 8 times,
                 reduced by 16 times, and reduced by 32 times.

    '''
    return resnet50()


if __name__ == '__main__':
    m= getBackBone("fcnResNet50")
    print(m)
    x = torch.rand((1, 3, 512, 512))
    print(x.shape)
    a, b, c, d = m(x)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    # show_model(m,"fcnResNet50")
