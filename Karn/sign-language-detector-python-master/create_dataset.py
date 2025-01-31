# import os
# import pickle

# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt


# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = './data'

# data = []
# labels = []
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []

#         x_ = []
#         y_ = []

#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y

#                     x_.append(x)
#                     y_.append(y)

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))

#             data.append(data_aux)
#             labels.append(dir_)

# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()


import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define data directory
DATA_DIR = './data'

data = []
labels = []

# Loop through each directory in the data folder
for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue  # Skip files, process only directories

    print(f"Processing class '{dir_}'...")
    
    # Loop through each image in the class directory
    for img_path in os.listdir(class_dir):
        data_aux = []
        x_ = []
        y_ = []

        img_full_path = os.path.join(class_dir, img_path)
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Could not read image {img_full_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = landmark.x
                    y = landmark.y

                    x_.append(x)
                    y_.append(y)

                # Normalize landmarks by translating them based on minimum x and y values
                min_x, min_y = min(x_), min(y_)
                for i, landmark in enumerate(hand_landmarks.landmark):
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)
                    
            data.append(data_aux)
            labels.append(dir_)
            print(f"Processed image: {img_full_path}")
        else:
            print(f"No hand landmarks detected in image {img_full_path}")

# Release Mediapipe resources
hands.close()

# Save data and labels using pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data successfully saved to 'data.pickle'")
