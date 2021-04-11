# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:19:34 2021

@author: Nicolas
"""
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

path = ".\\data\\part2\\database\\"
nb_same_image = 5
list_files = glob.glob(path+ "*.jpg")
#♦list_files = [file[len(path):-4] for file in list_files]
list_image = {}
list_histogramme = {}
for i, file in enumerate(list_files):
    file_name = file[len(path):-6]
    if(file_name not in list_image.keys()):
        list_image[file_name] = np.empty(nb_same_image, dtype=object)
        list_histogramme[file_name] = np.empty(nb_same_image, dtype=object)

for i, file in enumerate(list_files):
    file_name = file[len(path):-6]
    idx = int(file[-5])-1
    #idx = 4 - idx
 #   print(idx, file)
    im = cv2.imread(file)
    list_image[file_name][idx] = im
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

list_good_matches = {}
list_dist_max = {}
for name in list_image:
    i=1
    for image in list_image[name]:
        query_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        query_gray = cv2.cvtColor(query_image, cv2.COLOR_RGB2GRAY)
        query_keypoints, query_descriptor = orb.detectAndCompute(query_gray, None)
        keypoints_without_size = np.copy(query_image)
        cv2.drawKeypoints(query_image, query_keypoints, keypoints_without_size, color = (0, 255, 0))
        plt.imshow(keypoints_without_size, cmap='gray')
        plt.savefig("./results/p2/ORB/KP_train/KP_"+str(name)+"_"+str(i)+".jpg")
        i+=1
        plt.show()

