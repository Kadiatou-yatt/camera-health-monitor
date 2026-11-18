import cv2
import numpy as np
from src.quality import compute_quality

img = cv2.imread("images/test.jpg")
floue = cv2.GaussianBlur(img, (15, 15), 0)
sombre = (img * 0.3).astype(np.uint8)

for nom, im in [("originale", img), ("floue", floue), ("sombre", sombre)]:
    print(nom, compute_quality(im))