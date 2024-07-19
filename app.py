import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask,app,request,render_template
from tensorflow import keras
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras.models import Model


Model = keras.models.load_model('C:\\GIT\\Tomato-Plant-Disease-Recognition\\Training\\tmt.keras')
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/details')
def pred():
    return render_template('predict.html')

@app.route('/result',method=['GET','POST'])
def predict():
    if request.method =='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'Data','val','Tomato__healthy',f.filename)
        f.save(filepath)
        image= Image.open(filepath)
        image= np.asarray(image)
        pred = np.argmax(model.predict(image.reshape(-1,256,256,3)/255))
       

        classes = [ ' Tomato___Bacterial_spot ', 'Tomato___Early_blight ',
                    ' Tomato___healthy ' , 'Tomato___Late_blight ' ,
                    ' Tomato___Leaf_Mold' , 'Tomato___Septoria_leaf_spot ' ,
                    ' Tomato___Spider_mites Two-spotted_spider_mite ' ,
                    ' Tomato___Target_Spot ' , 'Tomato___Tomato_mosaic_virus ' ,
                    ' Tomato___Tomato_Yellow_Leaf_Curl_Virus ' ]
        keys = [ 'Tomato___Bacterial_spot ' , 'Tomato___Early_blight ' ,
                 ' Tomato___Late_blight ' , ' Tomato___Leaf_Mold' ,
                 ' Tomato___Septoria_leaf_spot ' , 'Tomato___Spider_mites Two-spotted_spider_mite ' ,
                 ' Tomato___Target_Spot ', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus ',
                 ' Tomato___Tomato_mosaic_virus ' , 'Tomato___healthy ']
        for j in list ( enumerate(keys) ) :
                if pred == j [0] :
                    prediction = j [1 ]


        if prediction == ' Tomato _Bacterial_spot' :
            return render_template( 'results.html ' , prediction_text = "to have Bacterial spots. ")
        elif prediction == 'Tomato _Early_blight ' :
            return render_template( 'results.html ' , prediction_text= "to have Early Blights . ")
        elif prediction == 'Tomato _healthy ' :
            return render_template( 'results.html ' , prediction_text = "to be healthy . ")
        elif prediction == 'Tomato _Late_blight ' :
            return render_template( 'results.html', prediction_text = "to have Late Blights . ")
        elif prediction == 'Tomato _Septoria_leaf_spot ' :
            return render_template( ' results.html ' , prediction_text = "to have Septoria_Leaf_Spot . ")
        elif prediction == 'Tomato Spider_mites Two- spotted spider_mite ' :
            return render_template( 'results.html ' , prediction_text = "to have Spider Mites. ")
        elif prediction == 'Tomato Target_Spot ' :
            return render_template( 'results.html ' , prediction_text = "to have Target spots . ")
        elif prediction == 'Tomato _Tomato_mosaic_virus ' :
            return render_template( 'results.html ' , prediction_text = "to have Tomato Mosaic Virus. ")
        elif prediction == 'Tomato _Leaf_Mold ' :
            return render_template( 'results.html ' , prediction_text = "to have Leaf Molds. ")
        elif prediction == 'Tomato _Tomato_Yellow_Leaf_Curl_Virus ' :
            return render_template( 'results.html ' , prediction_text = "to have Tomato Yellow Leaf Curl Virus. ")

        if __name__=="__main__":
            app.run(debug=True)    