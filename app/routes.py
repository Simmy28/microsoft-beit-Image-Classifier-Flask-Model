from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import os
from app.form import UploadForm  # Ensure you have this form defined
from app import app
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
import torch

# Load your BEiT model
save_dir = '/home/simran/workspace/project/checkpoint_200/best_model'
processor = AutoImageProcessor.from_pretrained(save_dir)
model = AutoModelForImageClassification.from_pretrained(save_dir)

# Define your classes
classes = ["Pattern", "Solid"]

# Define the expected input size for the model
width, height = 224, 224 

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height))  # Resize to fit model input size
    img_array = np.array(img).astype('float32')
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.transpose(img_array, (2, 0, 1))  # Reshape to (C, H, W)
    return img_array

import torch.nn.functional as F

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    img_tensor = torch.tensor(img_array).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    predicted_class = classes[predicted_class_id]
    
    print("Logits:", logits)
    print("Probabilities:", probabilities)
    # In the predict_image function, include:
    print(f"Predicted class: {predicted_class}, Probability: {probabilities[0][predicted_class_id]}")
    
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()  # Create form instance

    if form.validate_on_submit():
        image = form.image.data
        image_path = os.path.join(app.root_path, 'static', 'Images', image.filename)
        print("img path:", image_path)

        # Save the uploaded image
        image.save(image_path)

        # Make prediction
        prediction = predict_image(image_path)
        image_filename = os.path.join('Images', image.filename) # Get the filename after saving
        return render_template('index.html', prediction=prediction, image_path=image_filename , form=form)

    return render_template('index.html', form=form, image_path=None)  # Always pass the form object

if __name__ == '__main__':
    app.run(debug=True)