This is a short README for this project.


Run Tracker.py for detection, recognition and tracking. In Tracker.py, path of the video can be changed as you wish. Five testing video named videoX are provided in the zip package.

Note: the speed limit signs in the first few frames may not be detected, user need to press any key to start the tracking after the detected blue box showing up. User can also find the extract ROI and recognition result after detection works.

tracking_functions.py is about the functions implemented for tracking.

Detection.py is the detection function. Make sure the path of the classifier in it is correct.

SSIM.py is the recognition function.

SSIMdetection.py is for testing only the detection and recognition parts. User can test it with the images in “/Positives/*.png”.

“USspeedlimit” file stores the classifier cascade.xml

“speedmodels” file stores the standard five speed limit signs for
recognition.


