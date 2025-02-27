

from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import AdamW

import tensorflow as tf

import os
import json


"""
Author: Kumawat Mohit , Fulara Utkarsh
Date: 27-02-2025
Purpose: This file contains for loading the trained model and predict the class This should basically
be used if you do not want to try frontend and just play around from the controller and the code otherwise for
the api request we have a seperate file named GarbageClassifierAPI.py.
"""

def load_model_and_predict(image_path):
    """
    Loads the trained model and predicts the class of a given image.
    """
    # Load the trained model
    model = tf.keras.models.load_model('model_12_classes.keras')

    # Load class names from JSON file
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # Convert class indices back to a list
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]

    # Load & preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize

    # Predict class probabilities
    result = model.predict(img)
    predicted_index = np.argmax(result)

    # Get the predicted class name
    predicted_class = class_names[predicted_index]

    print(f" Predicted Class: {predicted_class}")



if __name__ == '__main__':
    print("Starting the script...")
    print(tf.__version__)
    sample_image_path = 'dataset/single_prediction/cloth.jpg'
    load_model_and_predict(sample_image_path)

    print("Script execution complete.")