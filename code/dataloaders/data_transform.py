from torchvision import transforms, utils
import torch


def basic_transform(img, bnd_boxes):
    
    image_transform = transforms.Compose([transforms.ToTensor()])
    bnd_transform = transforms.Compose([transforms.ToTensor()])

    bnd_boxes = torch.Tensor(bnd_boxes)

    img = image_transform(img)

    return img, bnd_boxes