from torchvision import transforms, utils
import torch

#TODO: Add various transforms to the data


def basic_transform(img, labels):
    
    image_transform = transforms.Compose([transforms.ToTensor()])
    bnd_transform = transforms.Compose([transforms.ToTensor()])

    labels = torch.LongTensor(labels)

    img = image_transform(img)
    
    return img, labels