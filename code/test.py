from dataloaders import RecieptDataset, basic_transform
from networks import East

root_dir = "E:\\Projects\\2020\\OCR\\EAST\\data\\processed_data"

dataset = RecieptDataset(root_dir, basic_transform)

img, bnd_box = dataset.__getitem__(0)

print(bnd_box.shape)