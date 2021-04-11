from tracker_particle import *

DIR_DATA_MOODLE = DATASET_FOLDER + "TP3_data/"
DIR_FRAMES = DIR_DATA_MOODLE + "frames/"
INIT_FILE = DIR_DATA_MOODLE + "init.txt"

init_coord = read_file(INIT_FILE)
mug1, mug2 = init_coord[0][2:6], init_coord[1][2:6]
print(f"mug1 = {mug1} (x1x2y1y2) = {x1x2y1y2_to_xywh(mug1)} (xywh) -> {xywh_to_x1x2y1y2(x1x2y1y2_to_xywh(mug1))}(x1x2y1y2) (check inverse)")
print(f"mug2 = {mug2} (x1x2y1y2) = {x1x2y1y2_to_xywh(mug2)} (xywh) -> {xywh_to_x1x2y1y2(x1x2y1y2_to_xywh(mug2))}(x1x2y1y2) (check inverse)")

mug1_bbox = x1x2y1y2_to_xywh(mug2)
mug2_bbox = x1x2y1y2_to_xywh(mug1)
frame_list = [DIR_FRAMES + 'frame' + str(k) + '.jpg' for k in range(1, 1011)]

tracker_particle(frame_list, [mug1_bbox, mug2_bbox])
