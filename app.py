import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, render_template, url_for, send_from_directory
import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet152V2
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from keras.models import Model

model = keras.models.load_model('Training/my_model.h5', custom_objects={'BatchNormalization': BatchNormalization})
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('results.html', prediction_text="No file part")

        f = request.files['image']
        if f.filename == '':
            return render_template('results.html', prediction_text="No selected file")

        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        # Load and preprocess the image
        try:
            image = Image.open(filepath)
            image = image.resize((256, 256))
            image = np.asarray(image)
            image = image.reshape(-1, 256, 256, 3) / 255.0

            # Predict the class of the image
            pred = np.argmax(model.predict(image))
            print(model.predict(image))

            classes = [
                'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot',
                'Tomato_Tomato_Yellow_Leaf_Curl_Virus',  'Tomato_Tomato_mosaic_virus', 'Tomato_healthy'
            ]

            prediction = classes[(pred) % 10]
            prediction_text = f"The tomato is predicted to have {prediction.replace('_', ' ').lower()}."
            return render_template('results.html', prediction_text=prediction_text)
        
        except Exception as e:
            return render_template('results.html', prediction_text=f"Error processing image: {str(e)}")
    
    return render_template('results.html', prediction_text="Invalid request method")

    
if __name__ == "__main__":
    app.run(debug=True)
