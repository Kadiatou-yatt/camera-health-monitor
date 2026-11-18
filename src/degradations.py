import cv2
import numpy as np


#  paliers de dégration entre 0 et 1 

def blur(image_bgr, level): #optique sale, flou bougé, mise au point perdue
    if level == 0:
        return image_bgr.copy()
    #calcul taille du noyau
    k = int(level* 30) + 1
    if k%2 == 0:
        k +=1 # noyau doit etre impair 
    return cv2.GaussianBlur(image_bgr, (k, k), 0) # calcul intensité flou à partir de la taille noyau
        
    
def darken(image_bgr, level) : #luminosité basse, tombée du jour
    if level == 0:
        return image_bgr.copy()
    #calculer facteur,  si palier : 1 => environ 10% de lumière restant
    factor = 1 - 0.9 * level 
    
    #conversion en deciaml 
    darker = image_bgr.astype(np.float32) * factor
    
    img_darken = np.clip(darker, 0, 255).astype(np.uint8)          
    return img_darken 

def distance(image_bgr, level): #personne plus loin
    if level == 0:
        return image_bgr.copy()
    
    facteur_reduction = 1 - 0.85 * level
    
    h, w = image_bgr.shape[:2] #retoune 
    new_w = int(w * facteur_reduction)
    new_h = int(h * facteur_reduction)
    
    #reduction image
    img_reduite = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    #Pose l’image réduite au centre d’un cadre de la taille d’origine
    top = (h - new_h) // 2  
    bottom = h - new_h - top 
    left = (w - new_w) // 2 
    right = w - new_w - left 
   
    return cv2.copyMakeBorder(img_reduite, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

DEGRADATIONS = {"blur": blur, "darken": darken, "distance": distance}
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]