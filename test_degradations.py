import cv2
from src.degradations import DEGRADATIONS, LEVELS
from src.detection import detect

img = cv2.imread("images/rue.jpg")
for name, fn in DEGRADATIONS.items():
    counts = [detect(fn(img, level))["count"] for level in LEVELS]
    print(name, counts)
    cv2.imwrite(f"images/deg_{name}.jpg", fn(img, 0.6))