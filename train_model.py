from utils.dataset_loader import load_fer2013_from_folders
from classifiers.emotion_model import build_emotion_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load dataset
train_gen, test_gen = load_fer2013_from_folders()

# Build model
model = build_emotion_model(input_shape=(48, 48, 1), num_classes=7)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    train_gen,
    epochs=25,
    validation_data=test_gen,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("models/emotion_cnn_fer2013.h5", save_best_only=True)
    ]
)