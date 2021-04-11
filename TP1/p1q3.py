import json
import cv2
import numpy as np
from matplotlib import pyplot as plt


def crop(image, start, end):
    return image[start[0]:end[0], start[1]:end[1]]

skate = []
path = r'.\\data\\part1\\images\\skate1.jpg'
skate.append(cv2.imread(path))
path = r'.\\data\\part1\\images\\skate2.jpg'
skate.append(cv2.imread(path))

with open("./data/part1/gt.json") as json_file:
    data = json.load(json_file)
    personne=[]
    for i in data['annotations']:
        if i['image'][:-1] == 'skate' and i['category_id']==1:
            start = (int(i['bbox'][0]),int(i['bbox'][1]))
            end = (int(i['bbox'][0]+i['bbox'][2]), int(i['bbox'][1]+i['bbox'][3]))
            personne.append(crop(skate[int(i['image'][-1])-1], start, end))

f, axes = plt.subplots(len(personne),2)
for j in range(len(personne)):
    color = ('b','g','r')
    hist = np.empty(3, dtype=object)
    for i, col in enumerate(color):
        hist[i] = cv2.calcHist([personne[j]],[i],None,[256],[0,256])
        axes[j][0].plot(hist[i], color = col)
    hsv = cv2.cvtColor(personne[j],cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    axes[j][1].plot(hist)


f.set_size_inches(20,8)
axes[0][0].title.set_text("Histogramme RGB de skate1.jpg")
axes[0][1].title.set_text("Histogramme HSV du skate1.jpg")
axes[1][0].title.set_text("Histogramme RGB du skate2.jpg")
axes[1][1].title.set_text("Histogramme HSV du skate2.jpg")
plt.savefig("./results/p1q3/histogrammes_skate.jpg")
plt.show()
