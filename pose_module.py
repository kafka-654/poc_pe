# Importing the necessary libraries
import cv2
import mediapipe as mp
import time
import math


#Creating a class poseDetector which will be used to create a mediapipe pose object and will contain all the functions,
#used in the main program

class poseDetector():
    #The arguments in the init function are the default arguments of the mediapipe pose object, we can change it
    # according to our needs.
    def __init__(self, mode=False,
               complexity=1,
               smoothness=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        # Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream.
        self.mode = mode
        # Complexity of the pose landmark model
        self.complexity = complexity
        # Whether to filter landmarks across different input images to reduce jitter
        self.smoothness = smoothness
        # Minimum confidence value ([0.0, 1.0]) for person detection to be considered successful.
        self.min_detection_confidence = min_detection_confidence
        # Minimum confidence value ([0.0, 1.0]) for the pose landmarks to be considered tracked successfully.
        self.min_tracking_confidence = min_tracking_confidence
        # Used for drawing the dots and the lines according to the landmarks on the image
        self.mp_draw = mp.solutions.drawing_utils
        # Initializing the pose object based on the above parameters
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.complexity, self.smoothness, self.min_detection_confidence,
                                      self.min_tracking_confidence)


    # Function for creating a pose object based on the image and drawing the landmarks spotted if the user wants it.
    # The function will return an image.
    def find_pose(self, img, draw = False):
        # Converting the image from bgr to rgb because mediapipe processes the image in rgb.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Getting the pose object of the image passed.
        self.pose_object = self.pose.process(img_rgb)
        # If the user sets the draw parameter to true, the function will plot all the landmarks detected along with
        # lines connecting them.
        if self.pose_object.pose_landmarks:
            if draw :
                self.mp_draw.draw_landmarks(img, self.pose_object.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img


    # In the find_pose function we have got the landmark points of the image but they are not stored in an accesible
    # way.
    # In the get_position function we store all the landmark points in the point_list. The point_list will store
    # the points as (l, x, y) where l will be the landmark number, x will be the x co-ordinate and y will be the y
    # co-ordinate.
    def get_position(self, img, draw = False):
        # Defining an empty point list
        self.point_list = []
        # The below lines will execute only if some landmarks are detected.
        if self.pose_object.pose_landmarks:
            for id, lm in enumerate(self.pose_object.pose_landmarks.landmark):
                h, w, c = img.shape
                # The landmarks are stored as ratios, we will covert them into x and y co-ordinates by multiplying
                # them with the corresponding width and height of the image.
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Appending them into a list.
                self.point_list.append([id, cx, cy])
                # If the draw is set to true, it will draw a circle along the detected points.
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            return self.point_list



    # This function will find the angle between any three landmark points and return it.
    def find_angle(self, img, p0, c, p1, draw=False):
        # The following lines will execute only if the point_list is not empty
        if len(self.point_list) != 0:

            # Getting the specified points from the point_list
            p0x, p0y = self.point_list[p0][1:]
            cx, cy = self.point_list[c][1:]
            p1x, p1y = self.point_list[p1][1:]

            try:
                # Calculating the angle between the points using the cosine formula
                p0c = math.sqrt(math.pow(cx-p0x, 2)+math.pow(cy-p0y, 2))
                p1c = math.sqrt(math.pow(cx-p1x, 2) + math.pow(cy - p1y, 2))
                p0p1 = math.sqrt(math.pow(p1x-p0x,2)+math.pow(p1y-p0y,2))
                angle = math.degrees(math.acos((p1c*p1c+p0c*p0c-p0p1*p0p1)/(2*p1c*p0c)))


                # If the draw is set to true, it will display the points and the angle on the image.
                if draw:
                    cv2.line(img, (p0x, p0y), (cx, cy), (0, 255, 0), 3)
                    cv2.line(img, (cx, cy), (p1x, p1y), (0, 255, 0), 3)
                    cv2.circle(img, (p0x, p0y), 3, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (p0x, p0y), 5, (255, 0, 255), 2)
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), 2)
                    cv2.circle(img, (p1x, p1y), 3, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (p1x, p1y), 5, (255, 0, 255), 2)
                    cv2.putText(img, str(int(angle)), (cx - 50, cy + 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 255), 1)

                return angle

            except :
                pass

    # Function to check if tadasan is detected in the image or not
    def tadasan_detection(self, img, draw = False):
        # Flag to check whether tadasan is detected or not , initially it is set to false
        flag = 0
        # Angle made by left hand
        angle_left = self.find_angle(img, 12, 14, 16, draw)
        # Angle made by right hand
        angle_right = self.find_angle(img, 11, 13, 15, draw)
        # Angle made by left leg
        angle_left_lower = self.find_angle(img, 23, 24, 26, draw)
        # Angle made by right leg
        angle_right_lower = self.find_angle(img, 25, 23, 24, draw)
        # Angle made by elbow, shoulder and waist
        angle_left_lower_1 = self.find_angle(img, 14, 12, 24, draw)
        angle_right_lower_1 = self.find_angle(img, 23,11,13, draw)

        # Checking the conditions for tadasan
        try:
            if angle_left > 140 and angle_right > 140 and 80 < angle_left_lower < 100 \
                    and 80 < angle_right_lower < 100 and angle_left_lower_1 > 150 \
                    and angle_right_lower_1 > 150:
                flag = 1
            else:
                pass
        # Type error occurs when we are not able to detect one of the above angles so a comparison happens between an
        # int type and a None type. In that case we just print a message that the tadasana was not detected.
        except TypeError:
            pass
        return img, flag

    # Function to check if bhujangasana is detected in the image or not
    def bhujangasana_detection(self, img, draw=False):
        # Flag to check whether bhujangasana is detected or not , initially it is set to false
        flag = 0
        # Angle made by the leg
        angle_leg = self.find_angle(img, 28, 26, 24, draw) #(172, 165, 173, 179)
        # Angle made by the waist
        angle_waist = self.find_angle(img, 26, 24, 12, draw) #(104, 121, 147, 109)
        # Angle made by hand
        angle_hand = self.find_angle(img, 16, 14, 12, draw) #(163, 135, 104, 164 )
        # Angle made by head
        angle_head = self.find_angle(img, 24, 12, 0, draw) #(154, 175, 158, 177)


        # Checking the conditions for bhujangasana
        try:
            if angle_leg > 150 and angle_waist < 150 and angle_hand < 170 and angle_head < 180:
                flag = 1
            else:
                pass
        # Type error occurs when we are not able to detect one of the above angles so a comparison happens between an
        # int type and a None type. In that case we just print a message that the tadasana was not detected.
        except TypeError:
            pass
        return img, flag


def main():
    # Capturing the video from the webcam using opencv.
    cap = cv2.VideoCapture(0)
    # prev_time is used for calculating the fps.
    prev_time = 0
    # Creating an object of the class poseDetector.
    detector = poseDetector()
    # Processing the video stream
    while True:
        # Calculation of the fps
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        # Reading the video image by image
        success, img = cap.read()
        # Getting the landmarks of the image using the find_pose function
        img = detector.find_pose(img, True)
        # Getting a list of landmark points using the get_position function
        point_list = detector.get_position(img)
        print(point_list)
        # Putting the fps on the image
        cv2.putText(img, str(int(fps)), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255),3)
        cv2.imshow("Image", img)
        # The waitkey is used to stop the execution for some time before each image. Here it is set to 1ms.
        cv2.waitKey(1)




if __name__ == "__main__":
    main()