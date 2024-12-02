import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load preprocessed data
X = np.load('landmarks.npy')
y = np.load('labels.npy')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
def build_model():
    model = models.Sequential([
        layers.Input(shape=(42,)),  # 21 landmarks * 2 (x, y)
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(26, activation='softmax')  # 26 letters A-Z
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('landmark_based_model.keras')