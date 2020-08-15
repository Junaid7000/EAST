from torchvision.ops import nms


#inbuilt nms for torchvision model
def torch_nms(boxes, scores, threshold = 0.5):
    return nms(boxes, scores, threshold)


#TODO: Add more efficient nms layer
#TODO: Add soft NMS layer
