import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViTImageProcessor, ViTForImageClassification, AutoModelForImageClassification, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification, DetrImageProcessor, DetrForObjectDetection
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

def predict_gibberish(text):
    model_name = "madhurjindal/autonlp-Gibberish-Detector-492513457"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)

    predicted_index = torch.argmax(probs, dim=1).item()
    predicted_prob = probs[0][predicted_index].item()
    labels = model.config.id2label
    predicted_label = labels[predicted_index]

    sorted_probs = sorted(
        [(labels[i], probs[0][i].item()) for i in range(len(labels))],
        key=lambda x: x[1],
        reverse=True
    )

    return predicted_label, predicted_prob, sorted_probs

def get_emotion(text):
    model_name = "mrm8488/t5-base-finetuned-emotion"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=2)
    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0].replace('<pad> ', '').replace('</s>', '')  # Cleaning up the output
    return label

def detect_objects(image):
    # Initialize the DETR model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

    # Process the image and perform object detection
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Retrieve detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detected_objects.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 3),
            "box": box
        })

    return detected_objects

st.title("Image and Text Classification App")

model_choice = st.selectbox("Select model", ["Age Classifier", "NSFW Detector", "Gibberish Detector", "Emotion Detector", "Object Detector"])

if model_choice == "Gibberish Detector" or model_choice == "Emotion Detector":
    text_input = st.text_area("Enter text")
    if text_input:
        try:
            if model_choice == "Gibberish Detector":
                predicted_label, predicted_prob, sorted_probs = predict_gibberish(text_input)
                st.write(f"Predicted Label: {predicted_label}")
                st.write(f"Prediction Probability: {predicted_prob:.4f}")

                st.write("Class Probabilities:")
                for label, proba in sorted_probs:
                    st.write(f"{label}: {proba:.4f}")
            else:
                emotion_label = get_emotion(text_input)
                st.write(f"Predicted Emotion: {emotion_label}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
elif model_choice == "Object Detector":
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

            detected_objects = detect_objects(image)
            for obj in detected_objects:
                st.write(f"Detected {obj['label']} with confidence {obj['score']} at location {obj['box']}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
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
            elif model_choice == "NSFW Detector":
                predicted_class, predicted_proba, sorted_probs = predict_nsfw(image)
                predicted_label = sorted_probs[0][0]
            else:
                st.write("Select a valid model")

            st.write(f"Predicted Class: {predicted_label}")
            st.write(f"Prediction Probability: {predicted_proba:.2%}")

            st.write("Class Probabilities:")
            for label, proba in sorted_probs:
                st.write(f"{label}: {proba:.2%}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
