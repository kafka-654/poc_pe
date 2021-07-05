import cv2
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


cap = cv2.VideoCapture(0)
prev_time = 0
while True:
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    prev_time = current_time
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose_object = pose.process(img_rgb)
    if pose_object.pose_landmarks:
        mp_draw.draw_landmarks(img, pose_object.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(pose_object.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id,lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255,0,0), cv2.FILLED)




    cv2.putText(img, str(int(fps)), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

