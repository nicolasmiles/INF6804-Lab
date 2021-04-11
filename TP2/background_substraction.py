import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tools_boxes

IMG_BASE_DIR = './/dataset//baseline//pedestrians//'
IMG_NAME = '000477'


def open_close_morpho(image, opening, closing_1, closing_2, print_image=False):
    # int dim
    image *= 255
    elem_structurant_ouverture = np.ones((opening, opening), np.uint8)
    elem_structurant_fermeture_1 = np.ones((closing_1, closing_1), np.uint8)
    elem_structurant_fermeture_2 = np.ones((closing_2, closing_2), np.uint8)
    image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, elem_structurant_fermeture_1)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, elem_structurant_ouverture)
    image_close2 = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, elem_structurant_fermeture_2)
    # Autre mask possible : MORPH_OPEN ,
    if print_image:
        plt.title("Image de la fermeture")
        plt.imshow(image_close2, vmin=0, vmax=255)
        plt.show()
    return image_close2


def detect(directory, img_name, plot_image=False, morph = True):
    directory += "input//"
    img_name = "in" + img_name + ".jpg"
    image1 = (cv2.imread(os.path.join(directory, 'in000001.jpg'), cv2.IMREAD_GRAYSCALE)).astype(float)
    image2 = (cv2.imread(os.path.join(directory, 'in000002.jpg'), cv2.IMREAD_GRAYSCALE)).astype(float)
    image3 = (cv2.imread(os.path.join(directory, 'in000003.jpg'), cv2.IMREAD_GRAYSCALE)).astype(float)
    image4 = (cv2.imread(os.path.join(directory, 'in000004.jpg'), cv2.IMREAD_GRAYSCALE)).astype(float)
    image5 = (cv2.imread(os.path.join(directory, 'in000005.jpg'), cv2.IMREAD_GRAYSCALE)).astype(float)

    Moy = (image1 + image2 + image3 + image4 + image5) / 5.0
    image_original = cv2.imread(os.path.join(directory, img_name))
    image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    image_detection = (cv2.imread(os.path.join(directory, img_name), cv2.IMREAD_GRAYSCALE)).astype(float)

    n = 40  # Ajuste la sensibilité de la détection
    foreground_image = np.abs(image_detection - Moy) > n
    if morph:
        image_seg = open_close_morpho(np.array(foreground_image, dtype=np.uint8), opening=4, closing_1=2, closing_2=20, print_image=False)
        boxes = tools_boxes.draw_box(image_original, image_seg, plot_image=plot_image)
    else:
        boxes = tools_boxes.draw_box(image_original, np.array(foreground_image, dtype=np.uint8), plot_image=plot_image)
    return boxes


if __name__ == "__main__":
    print(detect(IMG_BASE_DIR, IMG_NAME, plot_image=True))
