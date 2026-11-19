import cv2
import numpy as np

def to_gray(image_bgr): 
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray

def brightness(gray): 
    gray_mean_value = np.mean(gray)
    return float(gray_mean_value)

def sharpness(gray): 
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    gray_variance = np.var(laplacian)
    return float(gray_variance)

def contrast(gray):
    gray_std_value = np.std(gray)
    return float(gray_std_value)


def compute_quality(image_bgr): 
    img_gray = to_gray(image_bgr)
    
    gray_brightness = brightness(img_gray)
    gray_sharpness = sharpness(img_gray)
    gray_contrast = contrast(img_gray)
    
    img_quality = {"brightness": gray_brightness, "sharpness": gray_sharpness,"contrast": gray_contrast }
    
    return img_quality

# Seuils indicatifs, calibrés à la main sur mes images de test.
THRESHOLDS = {"dark": 40, "washed_out": 215, "low_contrast": 25, "blurry": 100}


def quality_verdict(q):
    # Traduit les trois indicateurs en verdict lisible, avec les raisons.
    b, s, c = q["brightness"], q["sharpness"], q["contrast"]

    # Image quasi noire : on s'arrete
    if b < THRESHOLDS["dark"] / 2:
        return "Inexploitable", ["Image quasi noire"]

    reasons = []
    if b < THRESHOLDS["dark"]:
        reasons.append("Image trop sombre")
    elif b > THRESHOLDS["washed_out"]:
        reasons.append("Image délavée ou surexposée")

    # Contraste et netteté baissent mécaniquement quand l'image s'assombrit.
    # On ne les juge que si la luminosité est correcte, sinon on compterait deux fois la même cause.
    if THRESHOLDS["dark"] <= b <= THRESHOLDS["washed_out"]:
        if c < THRESHOLDS["low_contrast"]:
            reasons.append("Contraste faible : voile, brume ou poussière possible")
        if s < THRESHOLDS["blurry"]:
            reasons.append("Netteté faible : flou probable")

    if not reasons:
        return "Exploitable", []
    if len(reasons) == 1:
        return "Limite", reasons
    return "Inexploitable", reasons