import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from path import Path
import tools_boxes

IMG_BASE_DIR = './/dataset//baseline//pedestrians//'
IMG_NAME = '000477'
THRESHOLD = 0.3  # The confidence score threshold to display a detection box

path = Path(".\\yolo\\")
net = cv2.dnn.readNetFromDarknet(path / 'yolov3-tiny.cfg', path / 'yolov3-tiny.weights')

detections_by_img_name = {}


def detect(directory, img_name, plot_image=False):
    directory += "input//"
    img_name = "in" + img_name + ".jpg"
    img = io.imread(os.path.join(directory, img_name))
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(ln)

    boxes = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > THRESHOLD:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height), class_id, confidence])
    if plot_image:
        fig, ax = plt.subplots(1)
        ax.imshow(img)
    index_to_remove = []
    for i in range(len(boxes)):
        x0_i, y0_i, width_i, height_i, _, _ = boxes[i]
        for j in range(i + 1, len(boxes)):
            x0_j, y0_j, width_j, height_j, _, _ = boxes[j]
            inter = np.sum(tools_boxes.inter((x0_i, y0_i, width_i, height_i), (x0_j, y0_j, width_j, height_j)))
            if inter != 0:
                x0_i = min(x0_i, x0_j)
                y0_i = min(y0_i, y0_j)
                width_i = max(x0_i + width_i, x0_j + width_j) - x0_i
                height_i = max(y0_i + height_i, y0_j + height_j) - y0_i
                index_to_remove.append(j)
        boxes[i] = [x0_i, y0_i, width_i, height_i]
    k = 0
    for j in index_to_remove:
        del boxes[j - k]
        k += 1
    if plot_image:
        for box in boxes:
            x0, y0, width, height = box
            rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor="k", facecolor='none')
            ax.add_patch(rect)
        # plt.savefig("results//office//yolo.jpg",bbox_inches='tight',pad_inches=0)
        plt.show()
    return boxes


if __name__ == "__main__":
    print(detect(IMG_BASE_DIR, IMG_NAME, plot_image=True))
