import cv2
import mediapipe as mp
import numpy as np
import glob
import os
import face_recognition
import os, sys
import cv2
import math
from deepface import DeepFace


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encodings = face_recognition.face_encodings(face_image)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
        print(self.known_face_names)

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        self.face_names = []

        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = 'Unknown'
            confidence = 'Unknown'
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])
            self.face_names.append(f'{name} ({confidence})')

        if self.face_locations:
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # rectagngle with names
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                # Admins Identification
                if "Dr. Momen.png" in name or "Dr. Ayman.png" in name:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0),
                                  2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0),
                                  -1)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0),
                                1)
                    admin_flag = 1  # 1 for admins
                else:
                    admin_flag = 0  # 0 lel non-admins

                # EMOTION DETECTION PARTTTT
                face_frame = frame[top:bottom, left:right]
                try:
                    results = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)
                    if results and isinstance(results, list):
                        dominant_emotion = results[0][
                            'dominant_emotion']
                    else:
                        dominant_emotion = "No dominant emotion"
                except Exception as e:
                    print("Error in emotion detection:", e)
                    dominant_emotion = "No emotion detected"

                cv2.putText(frame, dominant_emotion, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def load_and_resize_images(folder_path, size=(200, 200)):
    images = []
    image_names = []
    supported_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [file for ext in supported_extensions for file in glob.glob(os.path.join(folder_path, f'*{ext}'))]
    for img_path in image_files:
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, size)
            images.append(img)
            image_names.append(os.path.basename(img_path))
    return images, image_names

screen_width = 1000
screen_height = 800
grid_width = 200
grid_height = 200

ShoppingCart = []
count=0

table_image_path = r'ASSETS\table.jpg'
table_image = cv2.imread(table_image_path, cv2.IMREAD_UNCHANGED)

table_image = cv2.resize(table_image, (screen_width, screen_height))

folder_path = r"ASSETS\food"
images,image_names = load_and_resize_images(folder_path)
num_images_in_folder = len(images)


juice_folder_path = r"ASSETS\juice"
other_images,other_image_names = load_and_resize_images(juice_folder_path)
num_images_in_other_folder = len(other_images)

button_width = 50
button_height = screen_height

button_width2 = 50
button_height2 = screen_height

button_touched = False

selected_dish = -1
selected_dish_other_folder = -1

cap = cv2.VideoCapture(0)

remove_text_x = button_width // 2 - 10
remove_text_y = screen_height // 2 - 20
remove_text = "REMOVE"

next_text_x = screen_width - button_width2 // 2 - 10
next_text_y = screen_height//2 - 20
next_text = "NEXT"

finish_button_width = 75
finish_button_height = 50

finish_text_x = button_width + 25
finish_text_y = screen_height - 25
finish_text = "FINISH"

conveyor_speed = 2
offset = 0

face_recog = FaceRecognition()


while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    combined_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    combined_image = cv2.addWeighted(combined_image, 1, table_image, 0.8, 0)

    num_images = min(len(images), screen_width // grid_width)

    offset = (offset + conveyor_speed) % grid_width

    for i in range(num_images):
        img_index = (i + selected_dish + 1) % len(images) if selected_dish != -1 else i
        x_start = (i * grid_width + offset) % screen_width
        x_end = (x_start + grid_width) % screen_width
        if x_end > x_start:
            combined_image[:grid_height, x_start:x_end] = images[img_index]
        else:
            slice_width = screen_width - x_start
            combined_image[:grid_height, x_start:] = images[img_index][:, :slice_width]
            combined_image[:grid_height, :x_end] = images[img_index][:, slice_width:slice_width + x_end]

    face_recog.process_frame(image)


    for i in range(num_images):
        center_x = (i * grid_width + grid_width // 2 + offset) % screen_width
        center_y = grid_height // 2
        cv2.circle(combined_image, (center_x, center_y), 10, (0, 0, 255), -1)

    cv2.rectangle(combined_image, (0, 0), (button_width, button_height), (50, 50, 50), -1)

    cv2.rectangle(combined_image, (screen_width - button_width2, 0), (screen_width, screen_height), (50, 50, 50), -1)

    cv2.rectangle(combined_image, (button_width + 10, screen_height - finish_button_height),
                  (button_width + 10 + finish_button_width, screen_height),
                  (50, 50, 50), -1)


    for i, letter in enumerate(remove_text):
        cv2.putText(combined_image, letter, (remove_text_x, remove_text_y + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    for i, letter in enumerate(next_text):
        cv2.putText(combined_image, letter, (next_text_x, next_text_y + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(combined_image, finish_text, (finish_text_x, finish_text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(combined_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pointer_x = int(index_tip.x * screen_width)
            pointer_y = int(index_tip.y * screen_height)
            pointer_position = (pointer_x, pointer_y)

            cv2.circle(combined_image, pointer_position, 10, (0, 255, 0), -1)

            if pointer_x < button_width and selected_dish != -1:
                selected_dish = -1
            if pointer_x > screen_width - button_width and selected_dish != -1:
                if count==0:
                    selected_image_name = image_names[selected_dish]
                    ShoppingCart.append(selected_image_name)
                    selected_dish = -1
                    images = other_images
                    num_images = min(len(images), screen_width // grid_width)
                    count+=1
                elif count ==1:
                    selected_image_name = other_image_names[selected_dish]
                    ShoppingCart.append(selected_image_name)
                    selected_dish = -1
                    images = other_images
                    num_images = min(len(images), screen_width // grid_width)

            if pointer_x >button_width and pointer_x <button_width+20 and pointer_y > screen_height-finish_button_height:
                button_touched = True


            if button_touched:
                cart_items_without_extension = [os.path.splitext(item)[0] for item in ShoppingCart]
                order_text = "Your Order: " + ', '.join(cart_items_without_extension)
                cv2.putText(combined_image, order_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2,
                            cv2.LINE_AA)


            elif selected_dish == -1:
                for i in range(num_images):
                    center_x = (i * grid_width + grid_width // 2 + offset) % screen_width
                    center_y = grid_height // 2
                    distance = np.sqrt((pointer_position[0] - center_x) ** 2 + (pointer_position[1] - center_y) ** 2)
                    if distance < 50:
                        selected_dish = i
                        break

    if selected_dish != -1:
        dish_image = images[selected_dish]
        dish_x = screen_width // 2 - dish_image.shape[1] // 2
        dish_y = screen_height - dish_image.shape[0] - 50  # 50 pixel mn t7t
        combined_image[dish_y:dish_y + dish_image.shape[0], dish_x:dish_x + dish_image.shape[1]] = dish_image

    small_image = cv2.resize(image, (200, 150))
    combined_image[screen_height - 150:screen_height, screen_width - 200:screen_width] = small_image

    cv2.imshow("Interactive Display", combined_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Number of images in 'food' folder:", num_images_in_folder)

