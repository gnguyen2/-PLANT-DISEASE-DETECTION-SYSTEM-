from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
IMG_HEIGHT, IMG_WIDTH = 224, 224
model = load_model('model/plant_disease_model.h5')
class_names = [
    "bacterial_spot", "curl_virus", "early_blight", "healthy", "late_blight", "leaf_mold", "mosaic_virus"
]
# Define a function to prepare the image for prediction
def model_predict(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0,1]

    # Predict the class
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)
    return class_names[class_index[0]]  # Return the class name

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Prepare the image for prediction
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]

        return render_template('predict.html', uploaded_image_url=file_path, prediction=predicted_class_name)

if __name__ == '__main__':
    app.run()