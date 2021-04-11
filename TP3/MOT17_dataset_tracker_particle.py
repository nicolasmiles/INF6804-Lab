from tracker_particle import *

DIR_DATA_MOT = DATASET_FOLDER + "MOT17-11-FRCNN/"
DIR_FRAMES = DIR_DATA_MOT + "img1/"
INIT_FILE = DIR_DATA_MOT + "init.txt"

init_coord = read_file(INIT_FILE)
print(init_coord[0])
person1, person2 = init_coord[0][2:6], init_coord[1][2:6]

person1_bbox = person1
person2_bbox = person2
frame_list = glob.glob(DIR_FRAMES + "*.jpg")

tracker_particle(frame_list, person1_bbox, person2_bbox)
