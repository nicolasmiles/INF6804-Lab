import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random as ran
import time

DATASET_FOLDER = "./dataset/"
DIR_DATA_MOODLE = DATASET_FOLDER + "TP3_data/"
DIR_FRAMES = DIR_DATA_MOODLE + "frames/"
INIT_FILE = DIR_DATA_MOODLE + "init.txt"


def x1x2y1y2_to_xywh(coord):
    # coord = (x1, x2, y1, y2)
    return (coord[0], coord[2], abs(coord[1] - coord[0]), abs(coord[3] - coord[2]))


def xywh_to_x1x2y1y2(coord):
    # coord = (x, y, w, h)
    return (coord[0], coord[0] + coord[2], coord[1], coord[1] + coord[3])


def read_file(file_path):
    file = open(file_path, "r")
    lines = file.read().splitlines()
    for k in range(len(lines)):
        lines[k] = list(lines[k].split(" ", 5))
        for l in range(len(lines[k])):
            lines[k][l] = int(lines[k][l])
    return (lines)


init_coord = read_file(INIT_FILE)
tasse1, tasse2 = init_coord[0][2:6], init_coord[1][2:6]
print(f"tasse1 = {tasse1} (x1x2y1y2) = {x1x2y1y2_to_xywh(tasse1)} (xywh) -> {xywh_to_x1x2y1y2(x1x2y1y2_to_xywh(tasse1))}(x1x2y1y2) (check inverse)")
print(f"tasse2 = {tasse2} (x1x2y1y2) = {x1x2y1y2_to_xywh(tasse2)} (xywh) -> {xywh_to_x1x2y1y2(x1x2y1y2_to_xywh(tasse2))}(x1x2y1y2) (check inverse)")


def CompareParticles(hist1, hist2):
    res = 0
    for i in range(len(hist1)):
        res += 1 - cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_BHATTACHARYYA)
    return res / len(hist1)


def CalcHisto(image, bbox):
    roi = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.rectangle(mask, (roi[0], roi[1]), (roi[2], roi[3]), 255, -1, 8, 0)
    hist1 = cv2.calcHist([image], [0], mask, [64], [0, 256])
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.calcHist([image], [1], mask, [64], [0, 256])
    hist2 = cv2.normalize(hist2, hist2)
    hist3 = cv2.calcHist([image], [2], mask, [64], [0, 256])
    hist3 = cv2.normalize(hist3, hist3)
    return [hist1, hist2, hist3]


def generateParticles(particles, nb_particles, mvt=20, size_box=5, weight=None):
    if len(particles) == 1:
        # Pour générer les particules initiales. Pas de poids disponible
        # Déplacement aléatoire de ROI initial de [-mvt, mvt]
        new_particles = particles
        for i in range(1, nb_particles):
            part = [(particles[0][0] + ran.randint(-mvt, mvt), particles[0][1] + ran.randint(-mvt, mvt), particles[0][2] + ran.randint(-size_box, size_box), particles[0][3] + ran.randint(-size_box, size_box))]
            if i == 1:
                print("Première part = ", part)
            new_particles = new_particles + part
    else:
        # Échantillonage préférentiel avec la fonction ran.choices()
        temp = ran.choices(particles, weight)[0]
        # Mise à jour de leur état en ajoutant une translation en X et Y.
        part = [(temp[0] + ran.randint(-mvt, mvt), temp[1] + ran.randint(-mvt, mvt), temp[2] + ran.randint(-size_box, size_box), temp[3] + ran.randint(-size_box, size_box))]
        new_particles = part
        for i in range(1, nb_particles):
            temp = ran.choices(particles, weight)[0]
            part = [(temp[0] + ran.randint(-mvt, mvt), temp[1] + ran.randint(-mvt, mvt), temp[2] + ran.randint(-size_box, size_box), temp[3] + ran.randint(-size_box, size_box))]
            new_particles = new_particles + part
    return new_particles


image_name = 'frame1.jpg'

image1 = cv2.imread(DIR_FRAMES + 'frame1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(DIR_FRAMES + 'frame2.jpg', cv2.IMREAD_GRAYSCALE)

bbox1 = x1x2y1y2_to_xywh(tasse1)
bbox2 = x1x2y1y2_to_xywh(tasse2)


def pass_one_frame(img2, roi_hist, particles, nb_part=50, movement=100, size_box=50, plot_fig=0, color='r', plot_title="Suivi de la ROI", idx=0):
    # Calcule le suivi de l'objet identifié par init_coord entre les images img1 et img2
    # Mettre plot_fig=1 pour tout afficher / plot_fig=2 pour uniquement l'affichage sur img2
    # init_coord doit être x1x2y1y2
    # Renvoie les coordonnées x1x2y1y2 du ROI d'après

    particles = generateParticles(particles, nb_part, mvt=movement, size_box=size_box)

    # if plot_fig == 2:
    #     fig, ax = plt.subplots(1, figsize=(10, 10))
    #     ax.imshow(img1, cmap=plt.get_cmap('gray'))
    #     for p in particles:
    #         rect = patches.Rectangle((p[0], p[1]), p[2], p[3], linewidth=2, edgecolor=color, facecolor='none')
    #         ax.add_patch(rect)
    #     plt.show()

    weight = []
    for p in particles:
        candidate = CalcHisto(img2, p)
        dist = CompareParticles(roi_hist, candidate)
        weight.append(dist)

    p = particles[weight.index(max(weight))]

    # print(f"p={p} et next_coord={next_coord}")
    return p, particles


# Test:
# pass_one_frame(img1= image1, img2= image2, init_coord= tasse1, plot_fig=1, color='g')

scale = 0.25
box_size_evolution = 1
particle_movement = 20
nb_particles = 100

particle_red_mug = x1x2y1y2_to_xywh(tasse2)
particle_black_mug = x1x2y1y2_to_xywh(tasse1)
imageA = cv2.imread(DIR_FRAMES + 'frame' + str(1) + '.jpg', cv2.IMREAD_COLOR)
imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
height, width, depth = imageA.shape
imageA = cv2.resize(imageA, (int(width * scale), int(height * scale)))

particle_red_mug = tuple([int(item * scale) for item in particle_red_mug])
particles_list_red = [particle_red_mug]
roi_hist_red = CalcHisto(imageA, particle_red_mug)
particle_black_mug = tuple([int(item * scale) for item in particle_black_mug])
particles_list_black = [particle_black_mug]
roi_hist_black = CalcHisto(imageA, particle_black_mug)
start_time = time.time()
for k in range(2, 1010):
    # imageA = cv2.imread(DIR_FRAMES + 'frame' + str(k) + '.jpg', cv2.IMREAD_COLOR)
    # imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
    # imageA = cv2.resize(imageA, (int(width * scale), int(height * scale)))

    imageB = cv2.imread(DIR_FRAMES + 'frame' + str(k) + '.jpg', cv2.IMREAD_COLOR)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
    imageB = cv2.resize(imageB, (int(width * scale), int(height * scale)))
    title = "Suivi de la ROI sur l'image frame" + str(k) + ".png"
    # print(f"Pour k={k}, init_coord={p}")

    particle_red_mug, particles_list_red = pass_one_frame(idx=k, img2=imageB, roi_hist=roi_hist_red, particles=particles_list_red, size_box=box_size_evolution, movement=particle_movement, nb_part=nb_particles)
    particle_black_mug, particles_list_black = pass_one_frame(idx=k, img2=imageB, roi_hist=roi_hist_black, particles=particles_list_black, size_box=box_size_evolution, movement=particle_movement, nb_part=nb_particles)
    if k % 10 == 0:
        fig, ax = plt.subplots(1)
        ax.imshow(imageB)
        rect = patches.Rectangle((particle_red_mug[0], particle_red_mug[1]), particle_red_mug[2], particle_red_mug[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((particle_black_mug[0], particle_black_mug[1]), particle_black_mug[2], particle_black_mug[3], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        plt.title(title)
        plt.show()

end_time = time.time()
print(end_time - start_time)
