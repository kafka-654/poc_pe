
"""This program is used for the detection of yoga aasanas in videos"""

#Importing the necessary libraries
import cv2
import pose_module as pm
import time

# Creating an object of the poseDetector
detector = pm.poseDetector()
# Storing the video file in cap
cap = cv2.VideoCapture("videos/b1.mp4")
# Variable for storing the time
total_time = {"tadasana" : 0, "bhujangasana" : 0}

# Processing the video frame by frame
while True:
    # Time at the starting of a frame
    time1 = time.time()
    # Reading an image frame from cap
    success, img = cap.read()
    # This try loop is used to get out of the loop when the video finishes
    try:
        # Resizing the image for uniformity
        img = cv2.resize(img, (1280,720))
    except cv2.error:
        break
    # Finding the landmark points in that image
    img = detector.find_pose(img)
    # Storing the landmark points in the point_list
    point_list = detector.get_position(img)
    # Checking the image for tadasan
    img, flag_tadasana = detector.tadasan_detection(img)
    # Checking the image for bhujangasana, if bhujangasana is detected flag_bhujangasana becomes 1
    img, flag_bhujangasana = detector.bhujangasana_detection(img, True)
    # If tadasana is detected, the total_time["tadasana"] is incremented.
    if flag_tadasana == 1:
        time2 = time.time()
        total_time["tadasana"] += time2 - time1
        # Displaying tadasan and total time on screen
        cv2.putText(img, "Pose : Tadasana", (20, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, "Timer :"+str(int(total_time["tadasana"])), (1100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)

    # If tadasana is detected, the total_time["tadasana"] is incremented.
    elif flag_bhujangasana == 1:
        time2 = time.time()
        total_time["bhujangasana"] += time2 - time1
        # Displaying tadasan and total time on screen
        cv2.putText(img, "Pose : Bhujangasana", (20, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, "Timer :" + str(int(total_time["bhujangasana"])), (1100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

print("Tadasana : {} seconds".format(int(total_time["tadasana"])))
print("Bhujangasana : {} seconds".format(int(total_time["bhujangasana"])))
