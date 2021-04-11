import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

path = ".\\data\\part2\\database\\"
nb_same_image = 5
list_files = glob.glob(path+ "*.jpg")

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
    im = cv2.imread(file)
    list_image[file_name][idx] = im
    
def HistogramIntersection(vi, vj):
    inter = 0
    totVj = 0
    for k in range(len(vi)):
        inter += min(vi[k], vj[k])
        totVj += vj[k]
    return (inter / totVj)

for name in list_image:
    for idx, image in enumerate(list_image[name]):
        color = ('g', 'b', 'r')
        list_histogramme[name][idx] = np.empty((3,256), dtype=object)
        for i, col in enumerate(color):
            shape = image.shape
            list_histogramme[name][idx][i] = cv2.calcHist([image], [i], None, [256], [0, 256]).squeeze() / (shape[1] * shape[2])

for name in list_histogramme:
    mean = np.mean(list_histogramme[name], axis=0)
    list_histogramme[name] = np.append(np.empty(1, dtype=object), list_histogramme[name], axis=0)
    list_histogramme[name][0] = mean
    
    
#code pour tester l'ensemble d'entrainement avec le vecteur de test
# for i, file in enumerate(list_files):
#     file_name = file[len(path):-6]
#     idx = int(file[-5])
#     print("\n" +file[len(path):-4] + ": \n")
#     max_mean = 0
#     for name in list_histogramme:
#         moyenne = 0
#         for canal in range(3):
#             dist = HistogramIntersection(list_histogramme[name][0][canal], list_histogramme[file_name][idx][canal])
#             moyenne+= dist/3
#   #          print("Différence entre l'histogramme moyen de " + name+ " et l'image " + file[len(path):-4] + " pour le canal ", color[canal], " = ", dist)
#       #  print("Différence entre l'histogramme moyen de " + name+ " et l'image " + file[len(path):-4] + " = " + str(moyenne/3))
#         if moyenne>max_mean:
#             max_mean = moyenne
#             max_name = name
  #  print(file[len(path):-4]+ " est un(e) " + max_name + " ~ c'est " + ("vrai" if max_name == file_name else "faux") + ".")
   
 ## test des requetes  
path_query = ".\\data\\part2\\"

list_files_query = glob.glob(path_query+ "*.jpg")

list_image_query = {}
list_histogramme_query = {}

for i, file in enumerate(list_files_query):
    file_name = file[len(path_query):-4]
    im = cv2.imread(file)
    list_image_query[file_name] = im
for name in list_image_query:
    color = ('g', 'b', 'r')
    list_histogramme_query[name] = np.empty((3,256), dtype=object)
    image = list_image_query[name]
    for i, col in enumerate(color):
        shape = image.shape
        list_histogramme_query[name][i] = cv2.calcHist([image], [i], None, [256], [0, 256]).squeeze() / (shape[1] * shape[2])
for i, file in enumerate(list_files_query):
    file_name = file[len(path_query):-4]
    print("\n" +file_name + ": \n")
    max_mean = 0
    for name in list_histogramme:
        moyenne = 0
        for canal in range(3):
            dist = HistogramIntersection(list_histogramme[name][0][canal], list_histogramme_query[file_name][canal])
            moyenne+= dist/3
  #          print("Différence entre l'histogramme moyen de " + name+ " et l'image " + file[len(path):-4] + " pour le canal ", color[canal], " = ", dist)
      #  print("Différence entre l'histogramme moyen de " + name+ " et l'image " + file[len(path):-4] + " = " + str(moyenne/3))
        if moyenne>max_mean:
            max_mean = moyenne
            max_name = name
    print(file_name+ " est un(e) " + max_name + " ~ c'est " + ("vrai" if max_name == file_name[:-6] else "faux") + ".")
