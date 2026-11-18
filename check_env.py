import cv2

img = cv2.imread("images/test.jpg")
print(img.shape, img.dtype, img.min(), img.max())

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape, gray.mean())