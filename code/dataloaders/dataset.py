import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset


class RecieptDataset(Dataset):
    def __init__(self, root_dir, box_orientation = None, transform=None):
        super(RecieptDataset, self).__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.box_orientation = box_orientation

        #find all image files
        self.image_names = [path for path in os.listdir(self.root_dir) if path.endswith(".jpg")]

    def __len__(self):
        return len(self.image_names)

    #TODO: Read docs and paper on image transform and normalization
    def __getitem__(self, idx):
        
        img_name = self.image_names[idx]
        label_name = img_name[:-4]+".json"

        img = Image.open(os.path.join(self.root_dir, img_name))
        label_name = json.load(open(os.path.join(self.root_dir, label_name)))

        bnd_boxes = []
        for label in label_name:

            if self.box_orientation is not None:
                #For AABB boxes
                bnd_boxes.append([min(label['x']), min(label['y']), max(label['x']), max(label['y']), self.box_orientation])
            else:
                #TODO: Add support of RBOX
                bnd_boxes.append([min(label['x']), min(label['y']), max(label['x']), max(label['y'])])

        
        bnd_boxes = torch.Tensor(bnd_boxes)

        if self.transform is not None:
            img, bnd_boxes = self.transform(img, bnd_boxes)

        return [img, bnd_boxes]




if __name__ == "__main__":

    root_dir = "E:\\Projects\\2020\\OCR\\EAST\\data\\processed_data"

    dataset = RecieptDataset(root_dir, preprocessing_transform)

    img, bnd_box = dataset.__getitem__(0)

    print(type(img))

        



        
        

        