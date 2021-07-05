import streamlit as st
import cv2
import pose_module as pm
from PIL import Image
import numpy as np
import tempfile
import time

# Making a object of the poseDetector class from the pose module
detector = pm.poseDetector()

# Giving a title to the web app
st.title(" Yoga Pose Verification ")
# Making a sidebar for the webapp
st.sidebar.title(" Choose Option  ")

option = st.sidebar.selectbox("Media Type : ", ["Video", "Image"])

# If image verification is checked on the sidebar
if option == "Image":
    # Making a file uploader for an image
    file = st.file_uploader(label="Upload  ", type=["jpg", "jpeg", "cng", "webp","png" ], key="f1")
    if file:
        time1 = time.time()
        stframe_0 = st.empty()
        stframe_1 = st.empty()
        while True:
            pimage = Image.open(file)
            # st.image(pimage, width = 100)
            img = np.array(pimage)
            # st.write(image.shape)
            img = cv2.resize(img, (640, 480))
            # Finding the landmarks in the image
            img = detector.find_pose(img)
            # Getting a list of numbered landmarks in the image
            point_list = detector.get_position(img)
            # Checking the image for tadasana, if tadasana is detected flag_tadasana becomes 1
            img, flag_tadasana = detector.tadasan_detection(img, True)
            # Checking the image for bhujangasana, if bhujangasana is detected flag_bhujangasana becomes 1
            img, flag_bhujangasana = detector.bhujangasana_detection(img, True)
            if flag_tadasana == 1:
                cv2.putText(img, "Tadasana", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            elif flag_bhujangasana == 1:
                cv2.putText(img, "Bhujangasana", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            #cv2.imshow("Image", img)
            #cv2.waitKey(0)
            stframe_0.image(img)
            time2 =time.time()
            stframe_1.warning("Processing ...")
            # Refreshing the image for ten seconds
            if time2-time1 >=10:
                break

        if flag_tadasana == 1:
            stframe_1.info("Tadasana")
        elif flag_bhujangasana ==1:
            stframe_1.info("Bhujangasana")
        else:
            stframe_1.info("Not able to detect anything")
            #st.subheader("Not able to detect anything")


# If video detection is clicked
if option == "Video":
    f = st.file_uploader("Upload file")
    # Variable for storing the time
    total_time = {"tadasana": 0, "bhujangasana": 0}
    if f:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())

        cap = cv2.VideoCapture(tfile.name)


        stframe = st.empty()
        stframe_2 = st.empty()
        stframe_3 = st.empty()

        while True:
            # Time at the starting of a frame
            time1 = time.time()
            # Reading an image frame from cap
            success, img = cap.read()
            # This try loop is used to get out of the loop when the video finishes
            try:
                # Resizing the image for uniformity
                img = cv2.resize(img, (1280, 720))
            except cv2.error:
                break
            # Finding the landmark points in that image
            img = detector.find_pose(img)
            # Storing the landmark points in the point_list
            point_list = detector.get_position(img)
            # Checking the image for tadasan
            img, flag_tadasana = detector.tadasan_detection(img, True)
            # Checking the image for bhujangasana, if bhujangasana is detected flag_bhujangasana becomes 1
            img, flag_bhujangasana = detector.bhujangasana_detection(img, True)
            # If tadasana is detected, the total_time["tadasana"] is incremented.
            if flag_tadasana == 1:
                time2 = time.time()
                total_time["tadasana"] += time2 - time1
                # Displaying tadasan and total time on screen
                cv2.putText(img, "Pose : Tadasana", (20, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                cv2.putText(img, "Timer :" + str(int(total_time["tadasana"])), (1100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 0, 0), 4)

            # If tadasana is detected, the total_time["tadasana"] is incremented.
            elif flag_bhujangasana == 1:
                time2 = time.time()
                total_time["bhujangasana"] += time2 - time1
                # Displaying tadasan and total time on screen
                cv2.putText(img, "Pose : Bhujangasana", (20, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                cv2.putText(img, "Timer :" + str(int(total_time["bhujangasana"])), (900, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 0, 0), 4)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(img)
            stframe_2.warning("Processing ..")
        #st.write("Tadasana : {} seconds".format(int(total_time["tadasana"])))
        #st.write("Bhujangasana : {} seconds".format(int(total_time["bhujangasana"])))
        stframe_2.info("Tadasana : {} seconds".format(int(total_time["tadasana"])))

        stframe_3.info("Bhujangasana : {} seconds".format(int(total_time["bhujangasana"])))