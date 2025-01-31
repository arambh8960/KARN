# import os

# import cv2


# DATA_DIR = './data'
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# number_of_classes = 3
# dataset_size = 100

# cap = cv2.VideoCapture(2)
# for j in range(number_of_classes):
#     if not os.path.exists(os.path.join(DATA_DIR, str(j))):
#         os.makedirs(os.path.join(DATA_DIR, str(j)))

#     print('Collecting data for class {}'.format(j))

#     done = False
#     while True:
#         ret, frame = cap.read()
#         cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
#                     cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(25) == ord('q'):
#             break

#     counter = 0
#     while counter < dataset_size:
#         ret, frame = cap.read()
#         cv2.imshow('frame', frame)
#         cv2.waitKey(25)
#         cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

#         counter += 1

# cap.release()
# cv2.destroyAllWindows()







import os
import cv2

# Define constants
DATA_DIR = './data'
number_of_classes = 26
dataset_size = 100

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize the video capture; try different indices if necessary
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible. Please check the camera index or permissions.")
    cap.release()
    exit()

# Loop through each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for user input to start collecting data for the current class
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Display frame with instruction text
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Wait for 'q' key to proceed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Capture images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Display frame
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        # Save the captured frame to the specified directory
        file_path = os.path.join(class_dir, f'{counter}.jpg')
        success = cv2.imwrite(file_path, frame)
        if success:
            print(f"Image {counter} saved to {file_path}")
        else:
            print(f"Error: Failed to save image {counter}")

        counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()


