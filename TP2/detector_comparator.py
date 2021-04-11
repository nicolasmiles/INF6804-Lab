import background_substraction
import groundtruth_box_generator
import yolo_detector
import tools_boxes
import matplotlib.pyplot as plt
import numpy as np

# dataset range
# zoomInZoomOut = 500-815 (500-600) (512)
# pedestrians = 307-700 (307-400) (345)
# canoe = 845-1080 (845-1000) (891)
# office = 579-2043 (579-700) (620)
IMG_TYPE = 'office//'
IMG_BASE_DIR = './/dataset//' + IMG_TYPE
IMG_RANGE = [579, 700]
IMG_NAMES = [("000000" + str(i))[-6:] for i in range(IMG_RANGE[0], IMG_RANGE[1])]
score_bg = []
score_bg_without_morph = []
score_yolo = []
for img_name in IMG_NAMES:
    print(f"DEBUG : Image name = {img_name}")
    bg = background_substraction.detect(IMG_BASE_DIR, img_name, plot_image=False, morph=True)
    bg_without_morph = background_substraction.detect(IMG_BASE_DIR, img_name, plot_image=False, morph=False)
    gt = groundtruth_box_generator.detect(IMG_BASE_DIR, img_name, plot_image=False)
    yolo = yolo_detector.detect(IMG_BASE_DIR, img_name, plot_image=False)
    iou_yolo = tools_boxes.iou(gt, yolo)
    iou_bg = tools_boxes.iou(gt, bg)
    iou_bg_without_morph = tools_boxes.iou(gt, bg_without_morph)
    score_bg.append(iou_bg)
    score_bg_without_morph.append(iou_bg_without_morph)
    score_yolo.append(iou_yolo)

fig, ax = plt.subplots(1)

ax.plot(score_bg)
ax.plot(score_bg_without_morph)
ax.plot(score_yolo)
ax.legend(["Score Background", "Score Background without morphologic", "Score Yolo"])

plt.title("Score des méthodes par image")
plt.savefig(".//results//" + IMG_TYPE + "scores.png")
mean_bg = np.mean(score_bg)
mean_bg_without_morph = np.mean(score_bg_without_morph)
mean_yolo = np.mean(score_yolo)
print(f"Mean score background = {mean_bg / len(IMG_NAMES)}")
print(f"Mean score background without morphologic operations = {mean_bg_without_morph / len(IMG_NAMES)}")
print(f"Mean score Yolo = {mean_yolo / len(IMG_NAMES)}")
print(f"Ratio yolo/background = {mean_yolo / mean_bg}")
print(f"Ratio yolo/background without morphologic operations = {mean_yolo / mean_bg_without_morph}")

fichier = open(".//results//" + IMG_TYPE + "meanScore.txt", "w")
fichier.write(f"Résultat obtenu pour les images {IMG_TYPE[:-2]} allant de {IMG_RANGE[0]} à {IMG_RANGE[1]}\n")
fichier.write(f"Mean score background = {mean_bg / len(IMG_NAMES)}\n")
fichier.write(f"Mean score background without morphologic operations = {mean_bg_without_morph / len(IMG_NAMES)}\n")
fichier.write(f"Mean score Yolo = {mean_yolo / len(IMG_NAMES)}\n")
fichier.write(f"Ratio yolo/background = {mean_yolo / mean_bg}\n")
fichier.write(f"Ratio yolo/background without morphologic operations = {mean_yolo / mean_bg_without_morph}\n")
fichier.close()

plt.show()
