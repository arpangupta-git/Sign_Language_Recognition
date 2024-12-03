import os
import cv2
import numpy as np
import mediapipe as mp
import logging

# Suppress non-critical warnings from MediaPipe
logging.getLogger('mediapipe').setLevel(logging.ERROR)

def extract_landmarks_with_null_for_both_hands(dataset_path):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

    landmarks_data = []
    labels = []
    label_map = {letter: idx for idx, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    label_map["NULL"] = 26  # Adding null class

    for label in os.listdir(dataset_path):
        label_folder = os.path.join(dataset_path, label)
        if os.path.isdir(label_folder):
            for img_file in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Skipping invalid image: {img_path}")
                        continue

                    # Resize image and convert to RGB
                    img_resized = cv2.resize(img, (256, 256))  # Ensure a square input
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)

                    # Process detected hand landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Extract original landmarks
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                landmarks.append(lm.x)
                                landmarks.append(lm.y)
                            landmarks_data.append(landmarks)
                            labels.append(label_map[label])

                            # Create flipped landmarks (mirror effect for the other hand)
                            flipped_landmarks = []
                            for i in range(21):  # There are 21 landmarks for each hand
                                flipped_landmarks.append(1.0 - hand_landmarks.landmark[i].x)  # Flip x-coordinates
                                flipped_landmarks.append(hand_landmarks.landmark[i].y)  # y-coordinates stay the same

                            # Add flipped landmarks for the other hand
                            landmarks_data.append(flipped_landmarks)
                            labels.append(label_map[label])

                        print(f"Hand landmarks detected for {img_path}")
                    else:
                        # Handle NULL class explicitly
                        if label == "NULL":
                            landmarks_data.append([0.0] * 42)  # Default values for no detection
                            labels.append(label_map["NULL"])
                            print(f"No hand landmarks detected for NULL class: {img_path}")
                        else:
                            print(f"No hand landmarks detected for {img_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

    landmarks_data = np.array(landmarks_data)
    labels = np.array(labels)
    return landmarks_data, labels

# Path to dataset
dataset_path = 'C:\\Users\\arpan\\PycharmProjects\\Sign Language\\dataset'

# Extract landmarks and labels for both hands
X, y = extract_landmarks_with_null_for_both_hands(dataset_path)

# Save data for future use
np.save('landmarks_with_null_for_both_hands.npy', X)
np.save('labels_with_null_for_both_hands.npy', y)

# Print shapes of saved arrays
print(f"Landmarks shape: {X.shape}, Labels shape: {y.shape}")