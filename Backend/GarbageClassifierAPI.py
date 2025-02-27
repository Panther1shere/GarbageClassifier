from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import json
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

"""
Author: Kumawat Mohit , Fulara Utkarsh
Date: 27-02-2025
Purpose: This file contains the code for the FastAPI server that will serve the trained model.
"""

# Run the API server (use this command in terminal)
# uvicorn filename:app --host 0.0.0.0 --port 8000 --reload


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests from the frontend on localhost:4200
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Load the trained model
model = tf.keras.models.load_model('model_12_classes.keras')

# Load class indices from JSON
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Convert class indices back to a list
class_names = [class_indices[str(i)] for i in range(len(class_indices))]

# Define a function to preprocess the image received from the frontend
def preprocess_image(image_data):
    """Preprocesses the image for model prediction."""
    img = Image.open(BytesIO(image_data))
    img = img.resize((128, 128))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define the prediction endpoint with a POST request. The endpoint receives an image file and returns the predicted class.
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file, runs the model prediction, and returns the predicted class.
    """
    try:

        # Read image data
        image_data = await file.read()
        img = preprocess_image(image_data)

        # Predict class probabilities
        result = model.predict(img)
        predicted_index = np.argmax(result)
        predicted_class = class_names[predicted_index]

        return {"predicted_class": predicted_class}


    except Exception as e:
        return {"error": str(e)}


