import cv2
import numpy as np
from ultralytics import YOLO

_model = None

def load_model():
    global _model
    if  _model is None :
        _model = YOLO("yolov8n.pt")
    return _model

#Détecte les personnes dans une image et renvoie : combien, avec quelle confiance moyenne, et où (les boîtes)
def detect(image_bgr, conf_min=0.25, nms_iou=0.5):
    model = load_model()

    #classe[0]: classe personne de COCO 
    # [0] : YOLO renvoie une liste de résultats, un par image, ici on en envoie une seule.
    result = model.predict(image_bgr, classes=[0], conf=conf_min,  iou=nms_iou, verbose=False)[0]
    
    boxes = []
    for b in result.boxes : 
       boxes.append({
            "xyxy": b.xyxy[0].tolist(),  
            "conf": float(b.conf[0]),    
        })
    count = len(boxes)
    
    # Si personne n'est détecté, on renvoie 0 pour éviter une division par zéro
    mean_conf = sum(b["conf"] for b in boxes) / count if count > 0 else 0.0
    return {"count": count, "mean_conf": mean_conf, "boxes": boxes}
        

    
def draw_boxes(image_bgr, boxes):
    img_copy = image_bgr.copy()

    for b in boxes:                 
        # conversion en int pour openCV
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]

        # Rectangle : image, coin haut-gauche, coin bas-droite, couleur BGR, épaisseur.
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Confiance écrite au-dessus de la boîte.
        cv2.putText(
            img_copy,
            f"{b['conf']:.2f}",            
            (x1, max(y1 - 5, 15)),         #eviter déborder hors de l'image
            cv2.FONT_HERSHEY_SIMPLEX,     
            0.6,                          
            (0, 255, 0),                   
            2,                             
        )
    return img_copy