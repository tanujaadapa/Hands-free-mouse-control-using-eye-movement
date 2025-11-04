#This project uses the mediapipe library for facial landmark detection.
#Currently, mediapipe is not compatible with Python 3.13, so the project was implemented and tested successfully on Python 3.10.
import cv2
import mediapipe as mp
import pyautogui as pg

# Initialize webcam and FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen dimensions
screen_width, screen_height = pg.size()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    face_landmarks = results.multi_face_landmarks
    frame_height, frame_width, _ = frame.shape

    if face_landmarks:
        landmarks = face_landmarks[0].landmark

        # Iris landmarks (for cursor movement)
        for id, landmark in enumerate(landmarks[473:478]):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 2, (0, 255, 0))

            if id == 0:
                screen_x = screen_width / frame_width * x
                screen_y = screen_height / frame_height * y
                pg.moveTo(screen_x, screen_y)

        # Left eye landmarks (for blink detection)
        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 2, (0, 0, 255))

        # Blink to click
        if (left_eye[0].y - left_eye[1].y) < 0.008:
            pg.click()
            pg.sleep(0.3)

    cv2.imshow("cam capture", frame)
    cv2.waitKey(1)