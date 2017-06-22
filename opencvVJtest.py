import numpy as np
import cv2
import SSIM
import os

speed_limit = cv2.CascadeClassifier('USspeedlimit/cascade.xml')
#num_on_sign = cv2.CascadeClassifier('Speednumbers/cascade.xml')

# for loading the speed limit signs models
def load_images(folder):
       images = []
       for filename in os.listdir(folder):
              imgs = cv2.imread(os.path.join(folder,filename))
              if imgs is not None:
                     grayim = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
                     images.append(grayim)
       return images


img = cv2.imread('Positives/speedLimit35_1333393207.avi_image6.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, weight = img.shape[:2]
roi_gray = np.zeros((height, weight, 1), np.uint8)
#roi_gray_speed = np.zeros((height, weight, 1), np.uint8)
speed = speed_limit.detectMultiScale(gray)
for (x, y, w, h) in speed:
       img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = img[y:y+h, x:x+w]
       #num = num_on_sign.detectMultiScale(roi_gray)
       #for (sx, sy, sw, sh) in num:
              #cv2.rectangle(roi_gray, (sx, sy), (sx+sw, sy+sh), (0,255,0),2)
              #roi_gray_speed = roi_gray[sy:sy+sh, sx:sx+sw]


if cv2.countNonZero(roi_gray) == 0:
       print "None region of interest recognized"
#elif cv2.countNonZero(roi_gray_speed) == 0:
       #print "None region of speed number recognized"
       #cv2.imshow('img', img)
       #cv2.waitKey(0)
       #cv2.destroyAllWindows()

else:
       models = load_images('/Users/Lkid/PycharmProjects/openCVVJ/speedmodels')
       Smax = -2
       hroi, wroi = roi_gray.shape[:2]
       for m in models:
           m = cv2.resize(m, (wroi, hroi))
           _, meanS = SSIM.structual_similarity_ssim(roi_gray, m)
           print meanS
           if meanS > Smax:
                  Smax = meanS
                  sim = m
       print "Similarity is %.2f" % Smax
       cv2.imshow('roi_gray', roi_gray)
       cv2.imshow('sim', sim)
       cv2.imshow('img', img)
       cv2.waitKey(0)
       cv2.destroyAllWindows()

