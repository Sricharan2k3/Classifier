import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViTImageProcessor, ViTForImageClassification, AutoModelForImageClassification
import torch
import torch.nn.functional as F

# Define the id to label mapping for the age classifier
age_id2label = {
    "0": "0-2",
    "1": "3-9",
    "2": "10-19",
    "3": "20-29",
    "4": "30-39",
    "5": "40-49",
    "6": "50-59",
    "7": "60-69",
    "8": "more than 70"
}

def load_image(image_source):
    if isinstance(image_source, str):
        if image_source.startswith('http://') or image_source.startswith('https://'):
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_source)
    else:
        image = Image.open(image_source)
    return image

def predict_age(image):
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    processor = ViTImageProcessor.from_pretrained('nateraw/vit-age-classifier')

    inputs = processor(images=image, return_tensors='pt')
    output = model(**inputs)

    probs = torch.nn.functional.softmax(output.logits, dim=-1)
    predicted_class = probs.argmax(dim=-1).item()
    predicted_proba = probs[0, predicted_class].item()

    sorted_probs = sorted(
        [(age_id2label[str(i)], probs[0, i].item()) for i in range(len(age_id2label))],
        key=lambda x: x[1],
        reverse=True
    )

    return predicted_class, predicted_proba, sorted_probs

def predict_nsfw(image):
    processor = ViTImageProcessor.from_pretrained('AdamCodd/vit-base-nsfw-detector')
    model = AutoModelForImageClassification.from_pretrained('AdamCodd/vit-base-nsfw-detector')

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = model.config.id2label[predicted_class_idx]

    probabilities = F.softmax(logits, dim=-1)
    prediction_probability = probabilities[0][predicted_class_idx].item()

    sorted_probs = sorted(
        [(model.config.id2label[i], probabilities[0][i].item()) for i in range(len(model.config.id2label))],
        key=lambda x: x[1],
        reverse=True
    )

    return predicted_class_idx, prediction_probability, sorted_probs

st.title("Image Classification App")

model_choice = st.selectbox("Select model", ["Age Classifier", "NSFW Detector"])

upload_option = st.radio("Upload an image or enter an image URL", ("Upload", "URL"))

image = None  # Initialize image variable

if upload_option == "Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = load_image(uploaded_file)
else:
    image_url = st.text_input("Enter image URL")
    if image_url:
        image = load_image(image_url)

if image:
    try:
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if model_choice == "Age Classifier":
            predicted_class, predicted_proba, sorted_probs = predict_age(image)
            predicted_label = age_id2label[str(predicted_class)]
        else:
            predicted_class, predicted_proba, sorted_probs = predict_nsfw(image)
            predicted_label = sorted_probs[0][0]

        st.write(f"Predicted Class: {predicted_label}")
        st.write(f"Prediction Probability: {predicted_proba:.2%}")

        st.write("Class Probabilities:")
        for label, proba in sorted_probs:
            st.write(f"{label}: {proba:.2%}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
