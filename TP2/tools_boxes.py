import cv2
import matplotlib.pyplot as plt

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


def iou(ground_truth, comparison_method):
    iou = 0
    cnt = 1
    for j, box_sub in enumerate(comparison_method):
        cnt += 1
        for i, box_gt in enumerate(ground_truth):
            iou += same_windows(box_gt[0], box_gt[1], box_gt[2], box_gt[3], box_sub[0], box_sub[1], box_sub[2], box_sub[3])
    return iou/cnt


def draw_box(original, image, plot_image=False):
    copy = image.copy()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    points = []

    ROI_number = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ROI = image[y:y + h, x:x + w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)
        ROI_number += 1
        points.append([x, y, w, h])

    if plot_image:
        plt.imshow(original)
        plt.show()
    return points


if __name__ == "__main__":
    a = [[114, 58, 153, 121], [219, 39, 241, 121]]
    b = [[219, 39, 241, 121], [110, 42, 130, 108]]

    iou1 = iou(a, b)
    print(f"iou1 = {iou1}")
