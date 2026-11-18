import cv2
from src.detection import detect, draw_boxes



for name in ["test", "rue"]:
    img = cv2.imread(f"images/{name}.jpg")
    res = detect(img)
    print(name, res["count"], "personnes, confiance moyenne", round(res["mean_conf"], 2))
    cv2.imwrite(f"images/{name}_detections.jpg", draw_boxes(img, res["boxes"]))