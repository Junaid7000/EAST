import cv2
import os
import numpy as np
import json
from generate_score_geometry_map import create_geometry_map, create_score_map


def get_label_dict(label):

    label_dict = []

    for line in label:
        line = line.replace("\n", "").split(",")
        coords = line[:8]
        
        coords = [int(coord) for coord in coords]
        label = "".join(line[8:])

        coords_x = [x for idx, x in enumerate(coords) if idx%2==0]
        coords_y = [y for idx, y in enumerate(coords) if idx%2!=0]

        label_dict += [{"label": label, "x": coords_x, "y": coords_y}]
    
    return label_dict
        

def scale_image_and_bnd_boxes(image, label_dict, new_height, new_width):

    old_height, old_width , _ = image.shape

    if old_height > new_height:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    new_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    scale_X = new_width/old_width
    scale_Y = new_height/old_height

    for label in label_dict:
        label["x"] = (np.array(label['x'])*scale_X).astype(int).tolist()
        label["y"] = (np.array(label['y'])*scale_Y).astype(int).tolist()

    return new_image, label_dict


def adjust_bounding_box(label_dict, x1, y1, x_buffer, y_buffer):

    for label in label_dict:
        label["x"] = list(np.array(label['x']) - x1+x_buffer)
        label['y'] = list(np.array(label['y']) - y1+y_buffer)

    return label_dict


def reshape_large_images(image, label_dict):

    x_bnd = []
    y_bnd = []

    for label in label_dict:
        x_bnd += label['x']
        y_bnd += label['y']

    #find bounding box for the reciept
    xmax = np.max(x_bnd).item()
    ymax = np.max(y_bnd).item()
    xmin = np.min(x_bnd).item()
    ymin = np.min(y_bnd).item()

    #extra buffer whicle cropping.
    x_buffer = 70
    y_buffer = 20

    if x_buffer>xmin:
        x_buffer = xmin

    if y_buffer>ymin:
        y_buffer = ymin

    label_dict = adjust_bounding_box(label_dict, xmin, ymin, x_buffer, y_buffer)
    new_image = image[ymin-y_buffer:ymax+y_buffer, xmin-x_buffer:xmax+x_buffer]

    return label_dict, new_image


def save_label_dict(label_dict, save_path):
    with open(save_path, 'w+') as file:
        json.dump(label_dict, file)


def save_output_arrey(array, save_path):
    with open(save_path, 'wb') as file:
        np.save(file, array)

def reshape_image_and_bnd_box(image_folder, label_folder, save_folder):

    for i, image in enumerate(os.listdir(image_folder)):
        image_ = cv2.imread(os.path.join(image_folder ,image))
        label = open(os.path.join(label_folder, f"{image[:-4]}.txt")).readlines()
        # label_dict = get_label_dict(label)
        
        h, w, _ = image_.shape
        label_dict = get_label_dict(label)

        # if h>w and h<2500:
        #     image_, label_dict = scale_image_and_bnd_boxes(image_, label_dict, new_height, new_width)

        # else:

        label_dict, image_ = reshape_large_images(image_, label_dict)
        image_, label_dict = scale_image_and_bnd_boxes(image_, label_dict, new_height, new_width)
        #convert to single channel image
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)

        bnd_boxes = []
        for label in label_dict:
            bnd_boxes.append([min(label['x']), min(label['y']), max(label['x']), max(label['y'])])
        
        score = create_score_map(image_, bnd_boxes)
        geometry = create_geometry_map(image_, bnd_boxes)
        
        score = np.expand_dims(score, axis=0)
        output_array = np.concatenate([score, geometry], axis=0)

        print(output_array.shape)
        
        image_save_path = os.path.join(save_folder, image)
        array_save_path = os.path.join(save_folder, f"{image[:-4]}.npy")
        # label_save_path = os.path.join(save_folder, f"{image[:-4]}.json")
        # save_label_dict(label_dict, label_save_path)
        save_output_arrey(output_array, array_save_path)

        cv2.imwrite(image_save_path, image_)



if __name__ == "__main__":

    new_height = 1280
    new_width = 640

    image_folder = "./data/images"
    label_folder = "./data/labels"

    save_folder = "./data/processed_data/"

    reshape_image_and_bnd_box(image_folder, label_folder, save_folder)