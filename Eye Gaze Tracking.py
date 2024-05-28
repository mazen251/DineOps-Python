import cv2
import dlib
import numpy as np
import csv
import time

# Initialize dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to compute the gaze direction
def get_gaze_ratio(eye_points, facial_landmarks, frame, gray):
    # Getting the coordinates of the eye points
    left_eye_region = np.array([(facial_landmarks.part(eye_points[i]).x,
                                 facial_landmarks.part(eye_points[i]).y) for i in range(len(eye_points))], np.int32)

    # Calculate the height and width of the eye
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    eye = frame[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY), 70, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0: left_side_white = 1
    if right_side_white == 0: right_side_white = 1

    gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

# Starting the video capture
cap = cv2.VideoCapture(0)

# Create a CSV file for saving gaze direction and timestamps
csv_file = open("gaze_data.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Session', 'Gaze Direction'])

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Coordinates of the left and right eyes
        left_eye = [36, 37, 38, 39, 40, 41]
        right_eye = [42, 43, 44, 45, 46, 47]

        # Calculating the gaze ratio for both eyes
        gaze_ratio_left_eye = get_gaze_ratio(left_eye, landmarks, frame, gray)
        gaze_ratio_right_eye = get_gaze_ratio(right_eye, landmarks, frame, gray)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

        # Get the current timestamp
        timestamp = "Session Number 1"

        # Save the gaze direction and timestamp in the CSV file
        csv_writer.writerow([timestamp, gaze_ratio])

        # Detecting the gaze direction
        if gaze_ratio <= 0.9:
            cv2.putText(frame, "LOOKING RIGHT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        elif 1 < gaze_ratio < 2:
            cv2.putText(frame, "LOOKING LEFT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "LOOKING CENTER", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Close the CSV file
csv_file.close()

cap.release()
cv2.destroyAllWindows()
