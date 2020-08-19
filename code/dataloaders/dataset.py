import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RecieptDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(RecieptDataset, self).__init__()

        self.root_dir = root_dir
        self.transform = transform

        #find all image files
        self.image_names = [path for path in os.listdir(self.root_dir) if path.endswith(".jpg")]

    def __len__(self):
        return len(self.image_names)

    #TODO: Read docs and paper on image transform and normalization
    def __getitem__(self, idx):
        
        img_name = self.image_names[idx]
        label_name = img_name[:-4]+".npy"

        img = Image.open(os.path.join(self.root_dir, img_name))
        labels = np.load(open(os.path.join(self.root_dir, label_name), "rb"))
        
        if self.transform is not None:
            img, labels = self.transform(img, labels)

        return [img, labels]




if __name__ == "__main__":

    root_dir = "E:\\Projects\\2020\\OCR\\EAST\\data\\processed_data"

    dataset = RecieptDataset(root_dir, preprocessing_transform)

    img, bnd_box = dataset.__getitem__(0)

    print(type(img))

        



        
        

        