import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing import image



"""
Author: Kumawat Mohit , Fulara Utkarsh
Date: 27-02-2025
Purpose: This file contains the code. We have first learned to develop a model for distinguishing Cat and dog. 
Which when understood can be used to develop a model for garbage classification. Basically this is a very basic
code for understanding the working of CNN and how to develop a model for image classification. This is not a 
part of the orignal classifier and should only be read if you want to just get the fundamentals right. 
"""

def preprocess_data():
    """
    Loads and preprocesses training and test datasets.
    - Applies normalization to all images.
    - Augments training images for better generalization.
    """

    # Load training dataset from directory
    training_set = tf.keras.utils.image_dataset_from_directory(
        'dataset/training_set',
        image_size=(64, 64),
        batch_size=32,
        label_mode='binary'
    )

    print("Classes in Training Set:", training_set.class_names)

    # Normalize pixel values to the [0, 1] range
    normalization_layer = layers.Rescaling(1.0 / 255)
    training_set = training_set.map(lambda x, y: (normalization_layer(x), y))

    # Data Augmentation Pipeline (for training set only)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomContrast(0.2)
    ])

    # Apply data augmentation
    training_set = training_set.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Load test dataset (No augmentation, only normalization)
    test_set = tf.keras.utils.image_dataset_from_directory(
        'dataset/test_set',
        image_size=(64, 64),
        batch_size=32,
        label_mode='binary'
    )

    test_set = test_set.map(lambda x, y: (normalization_layer(x), y))

    return training_set, test_set


def build_cnn_network():
    """
    Builds a Convolutional Neural Network (CNN) model.
    Returns:
        - CNN model ready for training.
    """
    model = tf.keras.models.Sequential([
        # Convolutional + Pooling Layer 1
        layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
        layers.MaxPooling2D(pool_size=2, strides=2),

        # Convolutional + Pooling Layer 2
        layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),

        # Flattening
        layers.Flatten(),

        # Fully Connected Layer
        layers.Dense(units=128, activation='relu'),

        # Output Layer
        layers.Dense(units=1, activation='sigmoid')
    ])

    return model


def train_and_save_model():
    """
    Trains the CNN on the preprocessed dataset and saves the trained model.
    """
    # Load preprocessed data
    training_set, test_set = preprocess_data()

    # Build CNN model
    cnn = build_cnn_network()

    # Compile the CNN
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the CNN
    print("Training the CNN model...")
    cnn.fit(training_set, validation_data=test_set, epochs=25)

    # Save the trained model
    cnn.save('model.keras')
    print("Model saved as 'model.keras'.")


def load_model_and_predict(image_path):
    """
    Loads the trained model and predicts the class of a given image.
    Args:
        - image_path: Path to the image file to classify.
    """
    # Load the trained model
    model = tf.keras.models.load_model('model.keras')

    # Load and preprocess the image
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0) / 255.0  # Normalize

    # Predict class
    result = model.predict(test_image)

    # Output class
    predicted_class = "Dog" if result[0][0] > 0.5 else "Cat"
    print(f"Predicted Class: {predicted_class}")


if __name__ == '__main__':
    print("Starting the script...")

    # Train the model and save it
    #train_and_save_model()

    # Predict using a sample image
    sample_image_path = 'dataset/single_prediction/cat2.jpeg'  # Change if needed
    load_model_and_predict(sample_image_path)

    print("Script execution complete.")