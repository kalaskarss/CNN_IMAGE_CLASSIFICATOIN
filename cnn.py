import streamlit as st
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load VGG16 model without the top classification layer to use as a feature extractor
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Title of the app
st.title("Dog Breed Classifier with VGG16 Feature Extraction")

# Section to upload the pickled model file
model_file = st.file_uploader("Upload your pickled classifier (.pkl file)", type="pkl")

if model_file is not None:
    # Load the pickled model
    st.write("Loading model...")
    model = pickle.load(model_file)
    st.success("Model loaded successfully!")

    # Section to upload the image for classification
    uploaded_file = st.file_uploader("Upload an image to classify...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image to the required input shape
        st.write("Extracting features with VGG16...")
        image = image.resize((224, 224))  # Resize the image as required by VGG16
        image_array = img_to_array(image)  # Convert to array
        image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions to match VGG16 input
        image_array = preprocess_input(image_array)  # Preprocess for VGG16

        # Extract features using VGG16
        features = vgg_model.predict(image_array)
        features_flat = features.flatten().reshape(1, -1)  # Flatten to match model input shape (25088)

        # Predict the class using the loaded pickled model
        st.write("Classifying...")
        predictions = model.predict(features_flat)

        # Display the predictions (adjust the output format based on the model's return type)
        st.write("Predictions:", predictions)
