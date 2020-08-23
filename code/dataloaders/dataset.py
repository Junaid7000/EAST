import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .mask_generation import create_geometry_map, create_score_map

class RecieptDataset(Dataset):
    def __init__(self, root_dir, scale = 0.25 ,transform=None):
        super(RecieptDataset, self).__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale

        #find all image files
        self.image_names = [path for path in os.listdir(self.root_dir) if path.endswith(".jpg")]

    def __len__(self):
        return len(self.image_names)

    #TODO: Read docs and paper on image transform and normalization
    def __getitem__(self, idx):
        
        img_name = self.image_names[idx]
        label_name = img_name[:-4]+".json"

        img = Image.open(os.path.join(self.root_dir, img_name))
        labels_dict = json.load(open(os.path.join(self.root_dir, label_name), "rb"))

        score = create_score_map(img, labels_dict, self.scale)
        geo_map = create_geometry_map(img, labels_dict, self.scale)

        score = np.expand_dims(score, axis=0)
        output_array = np.concatenate([score, geo_map], axis=0)
        
        if self.transform:
            img, output_array = self.transform(img, output_array)

        return [img, output_array]




if __name__ == "__main__":

    root_dir = "E:\\Projects\\2020\\OCR\\EAST\\data\\processed_data"

    dataset = RecieptDataset(root_dir, preprocessing_transform)

    img, bnd_box = dataset.__getitem__(0)

    print(type(img))

        



        
        

        