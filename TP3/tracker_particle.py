import cv2

from utils import *


def tracker_particle(images_list, bbox, printing=True, scale=0.25, box_size_evolution=1, particle_movement=20, nb_particles=200, color_histogram=True, gradient_histogram=False, orb_descriptor=False):
    bbox1 = bbox[0]
    if len(bbox) > 1:
        bbox2 = bbox[1]

    if color_histogram:
        imageColor = cv2.imread(images_list[0], cv2.IMREAD_COLOR)
        imageColor = cv2.cvtColor(imageColor, cv2.COLOR_BGR2RGB)
        height, width, depth = imageColor.shape

    if gradient_histogram or orb_descriptor:
        imageGray = cv2.imread(images_list[0], cv2.IMREAD_GRAYSCALE)
        height, width = imageGray.shape
    imageGray = cv2.resize(imageGray, (int(width * scale), int(height * scale)))

    bbox1 = tuple([int(int(item) * scale) for item in bbox1])
    particles_list_red = [bbox1]
    if color_histogram and gradient_histogram:
        roi_red_color = computeColorHist(imageColor, bbox1)
        roi_red_gradient = computeGradientHist(imageGray, bbox1)
        roi_red = [roi_red_color, roi_red_gradient]
    elif color_histogram:
        roi_red = computeColorHist(imageColor, bbox1)
    elif gradient_histogram:
        roi_red = computeGradientHist(imageGray, bbox1)
    elif orb_descriptor:
        roi_red = compute_descriptor(imageGray, bbox1)
    if len(bbox) > 1:
        bbox2 = tuple([int(int(item) * scale) for item in bbox2])
        particles_list_black = [bbox2]
        if color_histogram and gradient_histogram:
            roi_black_color = computeColorHist(imageColor, bbox2)
            roi_black_gradient = computeGradientHist(imageGray, bbox2)
            roi_black = [roi_black_color, roi_black_gradient]
        elif color_histogram:
            roi_black = computeColorHist(imageColor, bbox2)
        elif gradient_histogram:
            roi_black = computeGradientHist(imageGray, bbox2)
        elif orb_descriptor:
            roi_black = compute_descriptor(imageGray, bbox2)
    start_time = time.time()
    bbox1_res = []
    for k in range(1, len(images_list)):
        if color_histogram:
            imageColor = cv2.imread(images_list[k], cv2.IMREAD_COLOR)
            imageColor = cv2.cvtColor(imageColor, cv2.COLOR_BGR2RGB)
            imageColor = cv2.resize(imageColor, (int(width * scale), int(height * scale)))
        if orb_descriptor or gradient_histogram:
            imageGray = cv2.imread(images_list[k], cv2.IMREAD_GRAYSCALE)
            imageGray = cv2.resize(imageGray, (int(width * scale), int(height * scale)))
        title = "Suivi de la ROI sur l'image frame" + str(k) + ".png"

        bbox1, particles_list_red = pass_one_frame(imgGray=imageGray, imgColor=imageColor, roi_gt=roi_red, particles=particles_list_red, size_box=box_size_evolution, movement=particle_movement, nb_part=nb_particles, color_histogram=color_histogram, gradient_histogram=gradient_histogram, orb_descriptor=orb_descriptor)
        bbox1_rescaled = tuple([int(int(item) / scale) for item in bbox1])
        bbox1_res.append(bbox1_rescaled)
        if len(bbox) > 1:
            bbox2, particles_list_black = pass_one_frame(imgGray=imageGray, imgColor=imageColor, roi_gt=roi_black, particles=particles_list_black, size_box=box_size_evolution, movement=particle_movement, nb_part=nb_particles, color_histogram=color_histogram, gradient_histogram=gradient_histogram, orb_descriptor=orb_descriptor)
        if printing and k % 10 == 0:
            fig, ax = plt.subplots(1)
            if color_histogram:
                ax.imshow(imageColor)
            else:
                ax.imshow(imageGray, cmap="gray")
            rect = patches.Rectangle((bbox1[0], bbox1[1]), bbox1[2], bbox1[3], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if len(bbox) > 1:
                rect = patches.Rectangle((bbox2[0], bbox2[1]), bbox2[2], bbox2[3], linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
            plt.title(title)
            plt.show()
        if k % 100 == 0:
            print(f"Step : {k}")
    end_time = time.time()
    print(end_time - start_time)
    return bbox1_res
