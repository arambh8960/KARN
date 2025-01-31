


# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']


# cap = cv2.VideoCapture(0)  # Change the index here if needed

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# while True:
#     data_aux = []
#     x_= []
#     y_= []
    
    
#     ret, frame = cap.read()
    
#     H, W, _ = frame.shape
#     if not ret:
#         print("Failed to grab frame")
#         break  # Exit the loop if frame capture fails
    
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
        
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
            
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x    
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))
#                 x.append(x)
#                 y.append(y)
        
        
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
        
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10
        
#         prediction = model.predict([np.asarray(data_aux)])        
#         predicted_character = labels_dict[int(prediction[0])] 
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)   
         
#     cv2.imshow('frame', frame)    
#     cv2.waitKey(1)
    
# cap.release()
# cv2.destroyAllWindows()


import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
#
# model_dict = pickle.load(open('./model.p', 'rb'))
model_dict = pickle.load(open('Karn\model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)  # Change to an accessible index if needed

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels dictionary
labels_dict = {0: 'Aa', 1: 'Bb', 2: 'Cc', 3: "Dd", 4: "Ee", 5: "Ff", 6: "Gg", 
               7: "Hh", 8: "Ii", 9: "Jj", 10: "Kk", 11: "Ll", 12: "Mm", 13: "Nn", 
               14: "Oo", 15: "Pp", 16: "Qq", 17: "Rr", 18: "Ss", 19: "Tt", 
               20: "Uu", 21: "Vv", 22: "Ww", 23: "Xx", 24: "Yy", 25: "Zz"}

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract and normalize landmark coordinates
            for i in range(21):  # Ensure exactly 21 landmarks
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            min_x, min_y = min(x_), min(y_)
            for i in range(21):
                x = hand_landmarks.landmark[i].x - min_x
                y = hand_landmarks.landmark[i].y - min_y
                data_aux.append(x)
                data_aux.append(y)

        # Check if data_aux has the correct number of features (42)
        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw a rectangle and display the predicted character
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        else:
            print("Error: Incorrect number of features in data_aux")

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



     