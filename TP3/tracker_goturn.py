from utils import *
init_coord = read_file(INIT_FILE)
tasse1, tasse2 = init_coord[0][2:6], init_coord[1][2:6]
print(f"tasse1 = {tasse1} (x1x2y1y2) = {x1x2y1y2_to_xywh(tasse1)} (xywh) -> {xywh_to_x1x2y1y2(x1x2y1y2_to_xywh(tasse1))}(x1x2y1y2) (check inverse)")
print(f"tasse2 = {tasse2} (x1x2y1y2) = {x1x2y1y2_to_xywh(tasse2)} (xywh) -> {xywh_to_x1x2y1y2(x1x2y1y2_to_xywh(tasse2))}(x1x2y1y2) (check inverse)")

image_name = 'frame1.jpg'

image1 = cv2.imread(DIR_FRAMES + 'frame1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(DIR_FRAMES + 'frame2.jpg', cv2.IMREAD_GRAYSCALE)

bbox1 = x1x2y1y2_to_xywh(tasse1)
bbox2 = x1x2y1y2_to_xywh(tasse2)

frame = cv2.imread(DIR_FRAMES + 'frame1.jpg', cv2.IMREAD_COLOR)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
scale = 1  # 0.25
height, width, depth = frame.shape
#frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

bbox = tuple([int(item * scale) for item in bbox2])

p1 = (int(bbox[0]), int(bbox[1]))
p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
plt.imshow(frame)
plt.show()

tracker = cv2.TrackerGOTURN_create()
ok = tracker.init(frame, bbox)

for i in range(2, 100):
    frame = cv2.imread(DIR_FRAMES + 'frame' + str(i) + '.jpg', cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    ok, bbox = tracker.update(frame)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    if i % 10 == 0:
        plt.imshow(frame)
        plt.show()
