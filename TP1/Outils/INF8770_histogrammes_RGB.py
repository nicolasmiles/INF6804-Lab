import numpy as np #pip install numpy==1.16.1
from matplotlib import pyplot as plt
import cv2
import os
import csv


pathData = 'C:/Users/anton/Desktop/Cours/INF8770 Multimedia/git/INF8770/TP3/data/'

""" CALCUL EN TROIS ETAPES : 
1 - calcul des histogrammes 
2 - stocker les histogrammes en mémoire
3 - faire les calculs de distance
"""

def histogram(image):
    color = ('b','g','r')
    hist = np.empty(3, dtype=object)
    for i, col in enumerate(color):
        hist[i] = cv2.calcHist([image],[i],None,[256],[0,256])
    return(hist)



def save_video_histogramme(pathFile):
    """parcours une frame par seconde d'une vidéo, et enregistre tous ses histogrammes. 
    Crée un fichier dans N lignes et 3 colonnes , N étant le nombre de secondes (et donc de frames sélectionnées)
    """
    video = cv2.VideoCapture(pathFile)
    numberFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) #Il faut donc créer numberFrames lignes dans le doc
    fps = video.get(cv2.CAP_PROP_FPS) #Le pas
    success,image = video.read()
    count = 0
    histFrames = np.empty(int(numberFrames/fps)+1, dtype=object)
    while success:
        histFrames[count] = histogram(image)
        compteur_temp = 0
        while success and compteur_temp < fps:
            success,image = video.read()
            compteur_temp += 1
        count += 1

    name_file = "histogramme_video/"+pathFile.replace(pathData, "").replace("video/","").replace(".mp4",".csv")

    file = open(name_file, "wb")
    np.save(file, histFrames)
    file.close
    return(1)


def read_histogram_file(pathFile):
    file = open(pathFile, "rb")
    hist = np.load(file, allow_pickle=True)
    return(hist)

def save_picture_histogramme(pathFile, imgType):
    img = cv2.imread(pathFile,cv2.IMREAD_COLOR)
    hist = histogram(img)
    name_file = "histogramme_image_"+str(imgType)+"/"+pathFile.replace(pathData, "").replace("jpeg/","").replace("png/","").replace(".png",".csv").replace(".jpeg",".csv")
    file = open(name_file, "wb")
    np.save(file, hist)
    file.close
    return(1)

def create_database(pathListImg, imgType, pathListVideo=None, printInfo=0):
    listImg =  os.listdir(pathListImg)
    for i, img in enumerate(listImg):
        image = pathListImg + img
        save_picture_histogramme(image, imgType)
        if printInfo==1 :
            print("Image ",img," traitée !")

    if pathListVideo is not None:
        listVid =  os.listdir(pathListVideo)
        for j, vid in enumerate(listVid):
            video = pathListVideo + vid
            save_video_histogramme(video)
            if printInfo==1 :
                print("Video ",vid," traitée !")

    return(1)


def find_Image_in_Video(pathImgNp, pathVideoNp):
    """ 
    Analyse une image (objet numpy) et une vidéo (objet numpy), renvoie la différence minimale et le timing où on a cette différence
    """
    img = read_histogram_file(pathImgNp)
    frames_video = read_histogram_file(pathVideoNp)
    diff = np.zeros(len(frames_video))
    for i, frame in enumerate(frames_video):
        if frame is not None:
            differance = abs(img-frame)
            differance_tot = differance[0].sum() + differance[1].sum() + differance[2].sum()
            diff[i]=differance_tot
        else:
            diff[i]=np.inf
    diff_min = diff.min()
    timing_ming = diff.argmin()
    return (diff_min, timing_ming)


def find_image_in_list_video(pathImgNp, pathListVideo):
    listVid =  os.listdir(pathListVideo)
    diff = np.zeros(len(listVid))
    timing = np.zeros(len(listVid))
    for j, vid in enumerate(listVid):
        video = pathListVideo + vid
        diff[j], timing[j] = find_Image_in_Video(pathImgNp, video)
    video_min = diff.argmin()
    timing_min = timing[video_min]
    return (video_min+1, timing_min)


def reconnaisance_image_video(pathListImg, pathListVideo, printInfo=0):
    listImg =  os.listdir(pathListImg)
    reconnaissance = np.empty(len(listImg), dtype='U25')
    for i, img in enumerate(listImg):
        image = pathListImg + img
        video_min, timing_min = find_image_in_list_video(image, pathListVideo)
        if len(str(video_min)) == 1:
            num_video="0"+str(video_min)
        else :
            num_video = str(video_min)
        result = str(img).replace(".csv","")+",v"+num_video+","+str(timing_min)
        if (printInfo==1):
            print(result)
        reconnaissance[i]=result

    return (reconnaissance)