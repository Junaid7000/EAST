import os
import cv2
import matplotlib.pyplot as plt


def plot_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def read_image_and_label(image, label):

    image = cv2.imread(image)
    
    label = open(label).readlines()
    
    plt.imshow(image)

    label_dict = {}
    for line in label:
        line = line.replace("\n", "").split(",")
        coords = line[:8]
        
        coords = [int(coord) for coord in coords]
        label = "".join(line[8:])

        coords_x = [x for idx, x in enumerate(coords) if idx%2==0]
        coords_y = [y for idx, y in enumerate(coords) if idx%2!=0]
        plt.plot(coords_x, coords_y)        
        
    plt.show()


if __name__ == "__main__":

    data = "data"
    image_folder = os.path.join(data, "images")
    label_folder = os.path.join(data, "labels")

    for image in os.listdir(image_folder):
        label = image[:-4] + ".txt"
        image = os.path.join(image_folder, image)
        label = os.path.join(label_folder, label)
        read_image_and_label(image, label)


    










