from tracking_functions import *
from Detection import *
import glob
import cv2

print "Loading image files from the video..."
Path = './/video5//'
imgList = []   # read all the video frames into the list
for i in glob.glob(Path + '*.png'):
    img = cv2.imread(i)
    imgList.append(img)
print "Finished loading images"

# determine the size of the video frames
VIDEO_HEIGHT, VIDEO_WIDTH = imgList[0].shape[:2]

# find the position of the object
i = 0
bB = detection(imgList[i])
while not bB:
    print "region of interest is empty, load next frame."
    cv2.imshow("nonROI", imgList[i])
    cv2.waitKey(200)
    i += 1
    bB = detection(imgList[i])

s_init = [bB[0], bB[1], 0, 0, float(bB[2])/float(bB[3]), bB[3]]

roiImg = imgList[i][bB[1]:(bB[1] + bB[3]), bB[0]:(bB[0] + bB[2])]
print "Find initial patch"

# initialize the color model
color_model = color_histogram(roiImg)

# initialize the particle set
N = 100
w_t = np.ones(N)/N
s_t = np.tile(s_init, (N, 1))

# Loop through the video and estimate the state at each step t
print "Start tracking..."
T = len(imgList)
for t in range(i, T):
    # randomly sample particles according to weights
    index = weighted_choice(w_t, N)
    s_t1 = np.zeros((100, 6), int)
    w_t1 = np.zeros(100, float)
    for m in range(0, N):
        s_t1[m] = s_t[int(index[m])]
        w_t1[m] = w_t[int(index[m])]
    s_t = s_t1
    w_t = w_t1

    # move particles according to the motion model
    s_t = motion_model(s_t, VIDEO_WIDTH, VIDEO_HEIGHT)

    # compute the appearance likelihood for each particle
    L = np.zeros(N)
    for l in range(0, N):
        L[l] = appearance_model(imgList[t], s_t[l], color_model, VIDEO_WIDTH, VIDEO_HEIGHT)
    # update particle weights
    for w in range(0, N):
        w_t[w] = w_t[w] * L[w]
    w_t = w_t/sum(w_t)
    # estimate the object location based on the particles
    estimate_t = np.dot(w_t, s_t)
    # draw particle filter estimate
    draw_box(imgList[t], estimate_t)
    cv2.imshow("Image", imgList[t])  # load the current image
    cv2.waitKey(200)

cv2.destroyAllWindows()
