import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random as ran
import time
import glob
from skimage.feature import hog

DATASET_FOLDER = "./dataset/"


def x1x2y1y2_to_xywh(coord):
    # coord = (x1, x2, y1, y2)
    return coord[0], coord[2], abs(coord[1] - coord[0]), abs(coord[3] - coord[2])


def xywh_to_x1x2y1y2(coord):
    # coord = (x, y, w, h)
    return coord[0], coord[0] + coord[2], coord[1], coord[1] + coord[3]


def read_file(file_path):
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
        for k in range(len(lines)):
            lines[k] = list(lines[k].split(" ", 5))
            for line in range(len(lines[k])):
                lines[k][line] = int(lines[k][line])

    return lines


def inter(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w <= 0 or h <= 0:
        return 0, 0, 0, 0
    return x, y, w, h


def same_windows(x1, y1, l1, h1, x2, y2, l2, h2):
    x, y, l, h = inter((x1, y1, l1, h1), (x2, y2, l2, h2))
    interS = l * h
    surface1 = l1 * h1
    surface2 = l2 * h2
    unionS = surface1 + surface2 - interS
    return interS / unionS


def iou(box_gt, box_sub):
    return same_windows(box_gt[0], box_gt[1], box_gt[2], box_gt[3], box_sub[0], box_sub[1], box_sub[2], box_sub[3])


def compareColorParticles(hist1, hist2):
    res = 0
    for i in range(len(hist1)):
        res += 1 - cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_BHATTACHARYYA)
    return res / len(hist1)


def compareGradientParticles(hist1, hist2):
    res = np.sum(hist1 == hist2)
    # res = np.sum(np.isclose(hist1, hist2))
    return res / (hist1.shape[0] * hist1.shape[1])


def computeColorHist(image, bbox):
    roi = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.rectangle(mask, (roi[0], roi[1]), (roi[2], roi[3]), 255, -1, 8, 0)
    hist1 = cv2.calcHist([image], [0], mask, [64], [0, 256])
    hist1 = cv2.normalize(hist1, hist1)
    # plt.plot(hist1)
    hist2 = cv2.calcHist([image], [1], mask, [64], [0, 256])
    hist2 = cv2.normalize(hist2, hist2)
    # plt.plot(hist2)
    hist3 = cv2.calcHist([image], [2], mask, [64], [0, 256])
    hist3 = cv2.normalize(hist3, hist3)
    # plt.plot(hist3)
    #  plt.show()
    return [hist1, hist2, hist3]


def computeGradientHist(image, bbox):
    roi = image[max(bbox[1], 0): min(bbox[1] + bbox[3], np.shape(image)[0]),
          max(bbox[0], 0): min(bbox[0] + bbox[2], np.shape(image)[1])]
    #if 0 in [max(bbox[1], 0), min(bbox[1] + bbox[3], np.shape(image)[0]),
    #      max(bbox[0], 0), min(bbox[0] + bbox[2], np.shape(image)[1])]:
    #    print([bbox[1], max(bbox[1], 0), bbox[1] + bbox[3], min(bbox[1] + bbox[3], np.shape(image)[0]),
    #      bbox[0], max(bbox[0], 0), bbox[0] + bbox[2], min(bbox[0] + bbox[2], np.shape(image)[1])])
    _, hog_image = hog(roi, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), visualize=True, multichannel=True)
    return hog_image


def generateParticles(particles, nb_particles, mvt=20, size_box=5, weight=None):
    if len(particles) == 1:
        # Pour générer les particules initiales. Pas de poids disponible
        # Déplacement aléatoire de ROI initial de [-mvt, mvt]
        new_particles = particles
        for i in range(1, nb_particles):
            part = [(particles[0][0] + ran.randint(-mvt, mvt), particles[0][1] + ran.randint(-mvt, mvt),
                     particles[0][2] + ran.randint(-size_box, size_box),
                     particles[0][3] + ran.randint(-size_box, size_box))]
            if i == 1:
                print("Première part = ", part)
            new_particles = new_particles + part
    else:
        # Échantillonage préférentiel avec la fonction ran.choices()
        temp = ran.choices(particles, weight)[0]
        # Mise à jour de leur état en ajoutant une translation en X et Y.
        part = [(temp[0] + ran.randint(-mvt, mvt), temp[1] + ran.randint(-mvt, mvt),
                 temp[2] + ran.randint(-size_box, size_box), temp[3] + ran.randint(-size_box, size_box))]
        new_particles = part
        for i in range(1, nb_particles):
            temp = ran.choices(particles, weight)[0]
            part = [(temp[0] + ran.randint(-mvt, mvt), temp[1] + ran.randint(-mvt, mvt),
                     temp[2] + ran.randint(-size_box, size_box), temp[3] + ran.randint(-size_box, size_box))]
            new_particles = new_particles + part
    return new_particles


def compute_descriptor(image, bbox):
    orb = cv2.ORB_create()
    image_gray = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    w, l = image_gray.shape
    if l > 1 and w > 1:
        _, descriptor = orb.detectAndCompute(image_gray, None)
    else:
        descriptor = None
    return descriptor


def comparison_ORB(query, train):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(query, train)
    matches = sorted(matches, key=lambda x: x.distance)

    score = 0
    n = 0
    weight = 2
    for mat in matches:
        score += 1 / (mat.distance + n * weight)
        n += 1
    # print(score)
    return score


def pass_one_frame(img, roi_gt, particles, nb_part=50, movement=100, size_box=50, color_histogram=True,
                   gradient_histogram=True):
    # Calcule le suivi de l'objet identifié par init_coord entre les images img1 et img2
    # Mettre plot_fig=1 pour tout afficher / plot_fig=2 pour uniquement l'affichage sur img2
    # init_coord doit être x1x2y1y2
    # Renvoie les coordonnées x1x2y1y2 du ROI d'après

    particles = generateParticles(particles, nb_part, mvt=movement, size_box=size_box)
    weight = []
    for p in particles:
        if color_histogram and gradient_histogram:
            roi_gt_color = roi_gt[0]
            roi_gt_gradient = roi_gt[1]
            candidate_color = computeColorHist(img, p)
            candidate_gradient = computeGradientHist(img, p)
            dist_color = compareColorParticles(roi_gt_color, candidate_color)
            dist_gradient = compareGradientParticles(roi_gt_gradient, candidate_gradient)
            dist = (dist_color + dist_gradient) / 2
        elif color_histogram:
            candidate = computeColorHist(img, p)
            dist = compareColorParticles(roi_gt, candidate)
        elif gradient_histogram:
            candidate = computeGradientHist(img, p)
            dist = compareGradientParticles(roi_gt, candidate)
        else:
            candidate = compute_descriptor(img, p)
            dist = comparison_ORB(roi_gt, candidate)
        weight.append(dist)
    p = particles[weight.index(max(weight))]

    return p, particles
