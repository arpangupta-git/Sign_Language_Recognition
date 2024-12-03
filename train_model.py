import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Load preprocessed data
print("Loading preprocessed data...")
X = np.load('landmarks_with_null_for_both_hands.npy')
y = np.load('labels_with_null_for_both_hands.npy')

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
def build_model():
    print("Building the model...")
    model = models.Sequential([
        layers.Input(shape=(42,)),  # Input: 21 landmarks * 2 (x, y)
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Prevent overfitting
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),  # Prevent overfitting
        layers.Dense(27, activation='softmax')  # Output: 26 letters + "NULL"
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize the model
model = build_model()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=30,  # Increased epochs for more robust training
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Save the trained model
model_path = 'landmark_based_model_with_null_for_both_hands.keras'
print(f"Saving the trained model to {model_path}...")
model.save(model_path)

print("Model training and saving completed successfully.")