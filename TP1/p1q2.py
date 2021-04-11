import json
import cv2
import numpy as np
from matplotlib import pyplot as plt


def crop(image, start, end):
    return image[start[0]:end[0], start[1]:end[1]]


path = r'.\\data\\part1\\images\\ski.jpg'
ski = cv2.imread(path)

with open("./data/part1/gt.json") as json_file:
    data = json.load(json_file)
    personne=[]
    for i in data['annotations']:
        if i['image'] == 'ski' and i['category_id']==1:
            start = (int(i['bbox'][0]),int(i['bbox'][1]))
            end = (int(i['bbox'][0]+i['bbox'][2]), int(i['bbox'][1]+i['bbox'][3]))
            personne.append(crop(ski, start, end))

f, axes = plt.subplots(1,len(personne))
f.set_size_inches(20,5)
for j in range(len(personne)):
    color = ('b','g','r')
    hist = np.empty(3, dtype=object)
    for i, col in enumerate(color):
        hist[i] = cv2.calcHist([personne[j]],[i],None,[256],[0,256])
        axes[j].plot(hist[i], color = col)

axes[0].title.set_text("Histogramme RGB du premier individu")
axes[1].title.set_text("Histogramme RGB du deuxième individu")
plt.savefig("./results/p1q2/histogramme.jpg")
plt.show()
