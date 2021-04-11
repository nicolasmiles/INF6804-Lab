import cv2
import os
import tools_boxes

IMG_BASE_DIR = './/dataset//baseline//pedestrians//'
IMG_NAME = '000401'


def detect(directory, img_name, plot_image=False):
    directory += "groundtruth//"
    img_name = "gt" + img_name + ".png"
    image_original = cv2.imread(os.path.join(directory, img_name))
    image_detection = cv2.imread(os.path.join(directory, img_name), cv2.IMREAD_GRAYSCALE)
    boxes = tools_boxes.draw_box(image_original, image_detection, plot_image=plot_image)
    return boxes


if __name__ == "__main__":
    print(detect(IMG_BASE_DIR, IMG_NAME, plot_image=True))
