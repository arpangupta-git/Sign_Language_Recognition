import cv2
import os

# Function to create directories for each letter (A-Z)
def create_directories(dataset_path):
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        letter_folder = os.path.join(dataset_path, letter)
        if not os.path.exists(letter_folder):
            os.makedirs(letter_folder)

# Function to capture images from webcam and save them to appropriate folder
def capture_images(dataset_path):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Create directories for A-Z
    create_directories(dataset_path)

    # Variable to store the current label
    current_label = 'A'
    image_count = 0

    print("Press 'Q' to quit, 'N' to go to next gesture.")
    print("Capture images for gesture: ", current_label)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Reverse the camera (flip the frame horizontally)
        frame = cv2.flip(frame, 1)  # 1 means horizontal flip (mirror effect)

        # Display the current gesture (label)
        cv2.putText(frame, f"Gesture: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Capture Your Dataset', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Save the image when 'S' is pressed
        if key == ord('s'):
            image_path = os.path.join(dataset_path, current_label, f"{current_label}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Captured {current_label}_{image_count}.jpg")
            image_count += 1

        # Move to next gesture when 'N' is pressed
        elif key == ord('n'):
            current_label = chr(ord(current_label) + 1) if current_label != 'Z' else 'A'
            image_count = 0
            print(f"Now capturing images for gesture: {current_label}")

        # Exit loop when 'Q' is pressed
        elif key == ord('q'):
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Path to save the dataset
dataset_path = 'C:\\Users\\arpan\\PycharmProjects\\Sign Language\\dataset'

# Start capturing images
capture_images(dataset_path)