import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

# Define the id to label mapping
id2label = {
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
    if image_source.startswith('http://') or image_source.startswith('https://'):
        response = requests.get(image_source)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_source)
    return image

def predict_image(image):
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    feature_extractor = ViTImageProcessor.from_pretrained('nateraw/vit-age-classifier')

    inputs = feature_extractor(images=image, return_tensors='pt')
    output = model(**inputs)

    probs = torch.nn.functional.softmax(output.logits, dim=-1)
    predicted_class = probs.argmax(dim=-1).item()
    predicted_proba = probs[0, predicted_class].item()

    sorted_probs = sorted(
        [(id2label[str(i)], probs[0, i].item()) for i in range(len(id2label))],
        key=lambda x: x[1],
        reverse=True
    )

    return predicted_class, predicted_proba, sorted_probs

st.title("Age Classification App")

image_url = st.text_input("Enter image URL")

if image_url:
    try:
        image = load_image(image_url)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        predicted_class, predicted_proba, sorted_probs = predict_image(image)
        predicted_label = id2label[str(predicted_class)]

        st.write(f"Predicted Age Group: {predicted_label}")
        st.write(f"Confidence: {predicted_proba:.4f}")

        st.write("Class Probabilities:")
        for label, proba in sorted_probs:
            st.write(f"{label}: {proba:.4f}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
