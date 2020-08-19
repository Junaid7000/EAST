import numpy as np


def get_all_coords(box):
    x = []
    y = []

    #coords are taken in clockwise direction
    x += [box[0], box[2], box[2], box[0]]
    y += [box[1], box[1], box[3], box[3]]

    return [x, y]


def scale_rectangle(box, scale):

    if type(box) == list:
        box = np.array(box)
    
    center = np.sum(box, axis=1)/4
    center = np.expand_dims(center, axis=1)
    cen_box = box - center
    cen_box = scale*cen_box

    box = cen_box + center

    return box.astype(int)


def map_score(score, box):
    h = box[1][3] - box[1][0]
    w = box[0][1] - box[0][0]

    score[box[1][0]:box[1][0]+h, box[0][0]:box[0][0]+w] = 1

    return score


def create_score_map(img, boxes):

    score = np.zeros_like(np.array(img))

    for box in boxes:
        box = get_all_coords(box)
        scaled_box = scale_rectangle(box, 0.7)
        score = map_score(score, scaled_box)

    return score


def calculate_geometry_score(box, indices, map_, point, idx = 0):
    
    h = box[1][3] - box[1][0]
    w = box[0][1] - box[0][0]

    geo = indices[:, box[1][0]:box[1][0]+h, box[0][0]:box[0][0]+w]
    geo = np.sqrt(np.square(point - geo[idx]))

    map_[box[1][0]:box[1][0]+h, box[0][0]:box[0][0]+w] = geo

    return map_


def create_geometry_map(img, boxes):

    left = np.zeros_like(np.array(img))
    right = np.zeros_like(np.array(img))
    top = np.zeros_like(np.array(img))
    bottom = np.zeros_like(np.array(img))
    orientation = np.zeros_like(np.array(img))

    indices = np.indices(np.array(img).shape)

    for box in boxes:
        box = get_all_coords(box)
        #left box[0][0]
        left = calculate_geometry_score(box, indices, left, box[0][0], 1)
        #right box[0][1]
        right = calculate_geometry_score(box, indices, right, box[0][1], 1)
        #top box[1][0]
        top = calculate_geometry_score(box, indices, top, box[1][0], 0)
        #bittom box[1][2]
        bottom = calculate_geometry_score(box, indices, bottom, box[1][2], 0)
    
    geometry_map = np.stack([left, top, right, bottom, orientation])

    return geometry_map