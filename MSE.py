import numpy as np
import cv2

# img_target = cv2.imread('speedmodels/speed25.png')
# img_model = cv2.imread('speedmodels/speed65.png')
#
# img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
# img_model = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)
#
# height, weight = img_target.shape[:2]
# img_model  = cv2.resize(img_model, (weight, height))

def mean_square_error(imageA, imageB):     #grayscale images
    err = np.sum((imageA.astype(np.float64)-imageB.astype(np.float64))**2)
    err /= float(imageA.shape[0]*imageA.shape[1])
    return err

def compare_images(imga,imgb):
    return 1/mean_square_error(imga,imgb)


