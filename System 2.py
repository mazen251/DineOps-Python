import cv2
import mediapipe as mp
import numpy as np
import glob
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def load_and_resize_images(folder_path, size=(200, 200)):
    images = []
    image_names = []
    supported_extensions = ('.jpg', '.jpeg', '.png')  # Add more extensions if needed
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
        dish_y = screen_height - dish_image.shape[0] - 50
        combined_image[dish_y:dish_y + dish_image.shape[0], dish_x:dish_x + dish_image.shape[1]] = dish_image

    small_image = cv2.resize(image, (200, 150))
    combined_image[screen_height - 150:screen_height, screen_width - 200:screen_width] = small_image

    cv2.imshow("Interactive Display", combined_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Number of images in 'food' folder:", num_images_in_folder)
