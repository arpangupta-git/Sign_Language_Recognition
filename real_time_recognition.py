import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model('landmark_based_model_with_null_for_both_hands.keras')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Label map for predictions
label_map = {idx: letter for idx, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
label_map[26] = "NULL"  # Null class

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set the desired frame rate (FPS)
desired_fps = 30
frame_time = 0

def is_open_hand(hand_landmarks):
    # Check if hand is open (fingers extended, no visible fist)
    distances = []
    palm_center = np.mean([[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y] for i in range(0, 5)], axis=0)
    for i in range(5, 21, 4):  # For the tips of the fingers (index, middle, ring, little)
        fingertip = [hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y]
        distance = np.linalg.norm(np.array(fingertip) - np.array(palm_center))
        distances.append(distance)

    return all(dist > 0.2 for dist in distances)  # Threshold may need tuning

while True:
    # Track frame time to control FPS
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)  # Horizontal flip to create a mirror effect

    # Get frame dimensions
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)

    letter = "Unknown"  # Default to "Unknown"
    confidence_score = 0  # Initialize confidence score

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Check if hand is open (e.g., fingers extended)
            if is_open_hand(hand_landmarks):
                letter = "Unknown"  # If the hand is open, classify as "Unknown"
                confidence_score = 100.0  # Fully confident that it's an open hand
            else:
                # Get bounding box coordinates for the hand
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                # Normalize landmarks within bounding box
                normalized_landmarks = []
                for lm in hand_landmarks.landmark:
                    normalized_x = (lm.x * w - x_min) / (x_max - x_min)
                    normalized_y = (lm.y * h - y_min) / (y_max - y_min)
                    normalized_landmarks.append([normalized_x, normalized_y])

                # Convert to numpy array
                landmarks = np.array(normalized_landmarks).reshape(1, -1)

                # Flip x-coordinates for the right hand
                if handedness.classification[0].label == "Right":
                    landmarks[0, ::2] = 1.0 - landmarks[0, ::2]  # Flip x-coordinates only for the right hand

                # Predict the gesture
                prediction = model.predict(landmarks)
                predicted_label = np.argmax(prediction)
                confidence_score = prediction[0][predicted_label] * 100  # Extract the confidence for the predicted label

                # Set a dynamic confidence threshold (e.g., 70%)
                if confidence_score > 70:  # Confidence threshold adjusted to 70%
                    letter = label_map[predicted_label]
                else:
                    letter = "Unknown"  # Label as "Unknown" if confidence is too low

            # Draw the bounding box with a unique color
            box_color = (0, 128, 255) if handedness.classification[0].label == "Left" else (128, 0, 255)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, 2)

            # Display the label and confidence just above the bounding box
            label_x = int(x_min)
            label_y = int(y_min) - 10  # Slightly above the box
            cv2.putText(frame, f'{letter} ({confidence_score:.2f}%)', (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculate FPS based on frame time
    frame_time = time.time() - start_time
    fps = int(1 / frame_time) if frame_time > 0 else 0

    # Limit frame rate to 30 FPS
    if frame_time < 1 / desired_fps:
        time.sleep(1 / desired_fps - frame_time)

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
