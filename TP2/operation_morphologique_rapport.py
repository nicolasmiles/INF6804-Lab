import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG_RESULT_DIR = ".//Rapport//operations_morphologiques//"
IMG_FILE = ".//Rapport//operations_morphologiques//Branche.jpg"
image = cv2.imread(IMG_FILE)
image2 = cv2.imread(IMG_FILE)

plt.title(f"Image")
plt.imshow(image, vmin=0, vmax=255)
plt.imsave(IMG_RESULT_DIR+"brancheplt.png", image)
plt.show()

print(image)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


opening = 3
closing = 3
elem_structurant_ouverture = np.ones((opening, opening), np.uint8)
elem_structurant_fermeture = np.ones((closing, closing), np.uint8)

image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, elem_structurant_ouverture)

plt.title(f"Fermeture de taille {closing}")
plt.imshow(image_close, vmin=0, vmax=255)
plt.imsave(IMG_RESULT_DIR+"Fermeture.png", image_close)
plt.show()

image_open = cv2.morphologyEx(image2, cv2.MORPH_OPEN, elem_structurant_ouverture)

plt.title(f"Ouverture de taille {opening}")
plt.imshow(image_open, vmin=0, vmax=255)
plt.imsave(IMG_RESULT_DIR+"Ouverture.png", image_open)
plt.show()
