
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau

import tensorflow as tf

import os
import json


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Gpu found successfully")


def preprocess_data():
    """
    Loads and preprocesses the dataset by splitting it into training (75%) & validation (25%) sets.
    Applies stronger **data augmentation**.
    """

    dataset_path = 'garbage_classification'

    # Data Augmentation Pipeline
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.3,
        shear_range=0.2,  # Added shear
        brightness_range=[0.5, 1.5],  # Brighter/darker images
        validation_split=0.25  # Splitting validation
    )

    # Load Training Set
    training_set = datagen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,  # Increased batch size
        class_mode="categorical",  # One-hot encoding
        subset="training",
        seed=42
    )


    # Load Validation Set (No Augmentation Here)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.25)
    validation_set = validation_datagen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
        seed=42
    )

    return training_set, validation_set, training_set.class_indices


def build_cnn_network():
    model = tf.keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation='swish', input_shape=(128, 128, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='swish', strides=2),

        layers.Conv2D(128, (3, 3), activation='swish'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), activation='swish'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='swish', strides=2),

        layers.Conv2D(256, (1, 1), strides=2, padding="same"),

        layers.Conv2D(512, (3, 3), activation='swish', padding="same"),
        layers.BatchNormalization(),


        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation='swish'),
        layers.Dropout(0.3),  # ✅ Reduced dropout for better regularization
        layers.Dense(12, activation='softmax')
    ])
    return model


def train_and_save_model():
    """
    Trains the CNN on the preprocessed dataset for 12 classes and saves the trained model.
    If a previously trained model exists, it loads the weights instead of training from scratch.
    """
    training_set, validation_set, class_indices = preprocess_data()
    print(f"Number of classes detected: {len(class_indices)}")

    # Build CNN model
    cnn = build_cnn_network()

    model_path = 'model_12_classes.keras'
    if os.path.exists(model_path):
        print("🔄 Loading existing model weights...")
        cnn.load_weights(model_path)

    cnn.compile(optimizer=AdamW(learning_rate=0.0005, weight_decay=5e-5),
                loss='categorical_crossentropy', metrics=['accuracy'])

    # Learning Rate Scheduler (Dynamically Adjusts Learning Rate)
    lr_scheduler = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-6)

    # Early Stopping (Avoids Overfitting)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    # Train the CNN Model
    print("🚀 Training the CNN model...")
    cnn.fit(training_set, validation_data=validation_set, epochs=50, callbacks=[early_stopping, lr_scheduler])

    # Save the trained model
    cnn.save(model_path)
    print(f"✅ Model saved as '{model_path}'.")


def load_model_and_predict(image_path):
    """
    Loads the trained model and predicts the class of a given image.
    """
    # ✅ Load the trained model
    model = tf.keras.models.load_model('model_12_classes.keras')


    # ✅ Load class names from JSON file
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # Convert class indices back to a list
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]

    # ✅ Load & preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize

    # ✅ Predict class probabilities
    result = model.predict(img)
    predicted_index = np.argmax(result)

    # ✅ Get the predicted class name
    predicted_class = class_names[predicted_index]

    print(f"✅ Predicted Class: {predicted_class}")


def SaveIndices(dataset_path):
    class_names = sorted(os.listdir(dataset_path))

    # Create a dictionary mapping indices to class names
    class_indices = {i: class_names[i] for i in range(len(class_names))}

    # Save to JSON file
    with open("class_indices.json", "w") as f:
        json.dump(class_indices, f)

    print("✅ Class indices saved successfully!")


if __name__ == '__main__':
    print("Starting the script...")
    print(tf.__version__)
    dataset_path = "garbage_classification"  # Change if your dataset is in another location
    #SaveIndices(dataset_path)

    # Get sorted list of class names from dataset folder

    # Train the model and save it
    #train_and_save_model()

    # Predict using a sample image
    sample_image_path = 'dataset/single_prediction/banana.jpeg.jpeg'
    load_model_and_predict(sample_image_path)

    print("Script execution complete.")