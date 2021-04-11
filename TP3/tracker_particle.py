from utils import *


def tracker_particle(images_list, bbox, printing=True, scale=0.25, box_size_evolution=1, particle_movement=20, nb_particles=200):
    bbox1 = bbox[0]
    if len(bbox) > 1:
        bbox2 = bbox[1]

    imageA = cv2.imread(images_list[0], cv2.IMREAD_COLOR)
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
    height, width, depth = imageA.shape
    imageA = cv2.resize(imageA, (int(width * scale), int(height * scale)))

    bbox1 = tuple([int(int(item) * scale) for item in bbox1])
    particles_list_red = [bbox1]
    roi_hist_red = computeHist(imageA, bbox1)
    if len(bbox) > 1:
        bbox2 = tuple([int(int(item) * scale) for item in bbox2])
        particles_list_black = [bbox2]
        roi_hist_black = computeHist(imageA, bbox2)
    start_time = time.time()
    bbox1_res = []
    for k in range(1, len(images_list)):
        # imageA = cv2.imread(DIR_FRAMES + 'frame' + str(k) + '.jpg', cv2.IMREAD_COLOR)
        # imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
        # imageA = cv2.resize(imageA, (int(width * scale), int(height * scale)))

        image = cv2.imread(images_list[k], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
        title = "Suivi de la ROI sur l'image frame" + str(k) + ".png"
        # print(f"Pour k={k}, init_coord={p}")

        bbox1, particles_list_red = pass_one_frame(idx=k, img=image, roi_hist=roi_hist_red, particles=particles_list_red, size_box=box_size_evolution, movement=particle_movement, nb_part=nb_particles)
        bbox1_rescaled = tuple([int(int(item) / scale) for item in bbox1])
        bbox1_res.append(bbox1_rescaled)
        if len(bbox) > 1:
            bbox2, particles_list_black = pass_one_frame(idx=k, img=image, roi_hist=roi_hist_black, particles=particles_list_black, size_box=box_size_evolution, movement=particle_movement, nb_part=nb_particles)
        if printing and k % 10 == 0:
            fig, ax = plt.subplots(1)
            ax.imshow(image)
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
