
"""This program is used for the detection of yoga aasanas in images."""

# Importing the necessary libraries and created modules
import cv2
import pose_module as pm
import time


# Creating an object of the poseDetector with the default parameters.
detector = pm.poseDetector()


# Processing the image
while True:
    # Reading the image
    img = cv2.imread("videos/b2img.png")
    # Resizing the image for uniformity
    img = cv2.resize(img, (640,480))
    # Finding the landmarks in the image
    img = detector.find_pose(img)
    # Getting a list of numbered landmarks in the image
    point_list = detector.get_position(img)
    # Checking the image for tadasana, if tadasana is detected flag_tadasana becomes 1
    img, flag_tadasana = detector.tadasan_detection(img)
    # Checking the image for bhujangasana, if bhujangasana is detected flag_bhujangasana becomes 1
    img, flag_bhujangasana = detector.bhujangasana_detection(img, True)
    if flag_tadasana == 1:
        cv2.putText(img, "Tadasana", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    elif flag_bhujangasana == 1:
        cv2.putText(img, "Bhujangasana", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    break

