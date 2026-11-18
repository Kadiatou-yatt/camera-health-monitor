import cv2
from src.degradations import blur, distance
from src.detection import detect, draw_boxes

img = cv2.imread("images/rue.jpg")
for name, im in [("blur_1", blur(img, 1.0)), ("distance_02", distance(img, 0.2))]:
    res = detect(im)
    print(name, res["count"])
    cv2.imwrite(f"images/check_{name}.jpg", draw_boxes(im, res["boxes"]))