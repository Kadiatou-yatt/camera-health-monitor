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