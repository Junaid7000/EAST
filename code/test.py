from dataloaders import RecieptDataset, basic_transform
from networks import East
from loss import ClassBalanceCrossEntropyLoss
from postprocessing import torch_nms

root_dir = "E:\\Projects\\2020\\OCR\\EAST\\data\\processed_data"

dataset = RecieptDataset(root_dir, transform =basic_transform)
# score_loss = ClassBalanceCrossEntropyLoss()
east = East(1, batch_norm = True)

img, bnd_box = dataset.__getitem__(0)
sco, out = east(img.unsqueeze(0))

# out= torch_nms(out, sco, 0.5)





print(img)