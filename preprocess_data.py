import os
import cv2
import numpy as np
import mediapipe as mp

def extract_landmarks(dataset_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)  # Increased confidence

    landmarks_data = []
    labels = []
    label_map = {letter: idx for idx, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}

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

                    # Resize the image for better hand detection
                    img_resized = cv2.resize(img, (640, 480))  # Resize to a fixed size
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)

                    # If hand landmarks are detected
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                landmarks.append(lm.x)
                                landmarks.append(lm.y)
                            landmarks_data.append(landmarks)
                            labels.append(label_map[label])
                    else:
                        print(f"No hand landmarks detected in: {img_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

    landmarks_data = np.array(landmarks_data)
    labels = np.array(labels)
    return landmarks_data, labels

# Path to dataset
dataset_path = 'C:\\Users\\arpan\\PycharmProjects\\Sign Language\\dataset'
X, y = extract_landmarks(dataset_path)

# Save the data for future use
np.save('landmarks.npy', X)
np.save('labels.npy', y)

print(f"Landmarks shape: {X.shape}, Labels shape: {y.shape}")