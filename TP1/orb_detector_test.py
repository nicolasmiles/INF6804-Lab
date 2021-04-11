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
list_dist_max["airplane"] = 80
list_dist_max["ball"] = 60
list_dist_max["car"] = 55
list_dist_max["cat"] = 90
list_dist_max["dolphin"] = 70
list_dist_max["face"] = 55
list_dist_max["lotus"] = 81
list_dist_max["strawberry"] = 60
for name in list_image:
    image_base = list_image[name][0]
    training_image = cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB)
    training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
    train_keypoints, train_descriptor = orb.detectAndCompute(training_gray, None)
    list_good_matches[name] = train_descriptor
    for image in list_image[name][1:]:
        query_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        query_gray = cv2.cvtColor(query_image, cv2.COLOR_RGB2GRAY)
        query_keypoints, query_descriptor = orb.detectAndCompute(query_gray, None)
        # matches = bf.knnMatch(query_descriptor, list_good_matches[name], k=2)
        matches = bf.match(query_descriptor, list_good_matches[name])
        matches = sorted(matches, key = lambda x:x.distance)
        good = []
        good = matches#[:300]
        # if(len(matches) == 0 or len(matches[0]) == 0):
        #       list_good_matches[name] = []
        #       break
        # for m,n in matches:
        #       if m.distance < 0.9*n.distance:
        #           good.append([m])
        # train_idxs = [mat[0].trainIdx for mat in good]
        dist_max = list_dist_max[name]#70#matches[50].distance
        keypoints_without_size = np.copy(query_image)
        query_keypoints = np.array(query_keypoints)
        train_idxs = [mat.trainIdx for mat in good if mat.distance < dist_max]
        query_idxs = [mat.queryIdx for mat in good if mat.distance < dist_max]
        cv2.drawKeypoints(query_image, query_keypoints[query_idxs], keypoints_without_size, color = (0, 255, 0))
        plt.imshow(keypoints_without_size, cmap='gray')
        plt.show()

        # query_idxs = [mat[0].queryIdx for mat in good]
        list_good_matches[name] = np.append(list_good_matches[name][train_idxs], query_descriptor[query_idxs], axis=0)
      #  list_dist_max[name] = dist_max
def comparaison_ORB(im_query, good_matches):
    query_image = cv2.cvtColor(im_query, cv2.COLOR_BGR2RGB)
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_RGB2GRAY)
    query_keypoints, query_descriptor = orb.detectAndCompute(query_gray, None)
   # matches = bf.knnMatch(query_descriptor, good_matches, k=2)
    matches = bf.match(query_descriptor,list_good_matches[name])
    good = []
    good = matches
    # if(len(matches) == 0 or len(matches[0]) == 0):
    #     list_good_matches[name] = []
    # for m,n in matches:
    #     if m.distance < 0.9*n.distance:
    #         good.append([m])
    dist_max = list_dist_max[name]
    query_idxs = [mat.queryIdx for mat in good if mat.distance < dist_max]
    good = query_descriptor[query_idxs]
    print(len(good))
    print(len(good_matches))
    return len(good)/len(good_matches)
path_query = ".\\data\\part2\\"

list_files_query = glob.glob(path_query+ "*.jpg")
#♦list_files = [file[len(path):-4] for file in list_files]

for i, file in enumerate(list_files_query):
    file_name = file[len(path_query):-4]
    im_query = cv2.imread(file)
    print("\n" +file_name + ": \n")
    max_matches = 0
    max_name = "echec"
    for name in list_good_matches:
        matches = comparaison_ORB(im_query, list_good_matches[name])
        if matches>max_matches:
            max_matches = matches
            max_name = name
    print(file_name+ " est un(e) " + max_name + " ~ c'est " + ("vrai" if max_name == file_name[:-6] else "faux") + ".")

 