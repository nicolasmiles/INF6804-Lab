from tracker_particle import *
from evolutionary_search import maximize

DIR_DATA_MOT = DATASET_FOLDER + "MOT17-11-FRCNN/"
DIR_FRAMES = DIR_DATA_MOT + "img1/"
INIT_FILE = DIR_DATA_MOT + "gt/gt.txt"

with open(INIT_FILE, "r") as file:
    lines = file.read().splitlines()[:900]
gt = np.array([item.split(',')[2:6] for item in lines], dtype=int)
person1_bbox = gt[0]

frame_list = glob.glob(DIR_FRAMES + "*.jpg")[:900]


def track_MOT(scale=0.25, box_size_evolution=1, particle_movement=20, nb_particles=200):
    bbox_list = tracker_particle(frame_list, [person1_bbox], printing=False,
                                 scale=scale, box_size_evolution=box_size_evolution, particle_movement=particle_movement, nb_particles=nb_particles)

    accuracy = 0
    for i, bbox in enumerate(bbox_list):
        res = iou(gt[i], bbox)
        if res > 0.5:
            accuracy += 1

    accuracy /= len(bbox_list)
    return accuracy


param_grid = {'scale': np.arange(0.1, 0.25, step=0.05), 'box_size_evolution': range(1, 20, 2),
              'particle_movement': range(1, 50, 2), 'nb_particles': np.append(range(1, 100, 10), range(100, 500, 50))}
args = {}
best_params, best_score, score_results, _, _ = maximize(track_MOT, param_grid, args, verbose=True)
print(best_params)
print(best_score)
print(score_results)
