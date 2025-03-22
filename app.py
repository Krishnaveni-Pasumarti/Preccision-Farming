
from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import cv2 as cv
import base64
import keras

app = Flask(__name__)

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# Load the trained machine learning models
with open('C:/Users/parvathi143/Desktop/webapp/webapp/models/RandomForest.pkl', 'rb') as recommendation_model_file:
    recommendation_model = pickle.load(recommendation_model_file)

with open('C:/Users/parvathi143/Desktop/webapp/webapp/models/RandomForest_Yieldd.pkl', 'rb') as yield_model_file:
    yield_model = pickle.load(yield_model_file)
with open('C:/Users/parvathi143/Desktop/webapp/webapp/models/fertilizer_model.pkl', 'rb') as ferti_model_file:
    ferti = pickle.load(ferti_model_file)

model = pickle.load(open('C:/Users/parvathi143/Desktop/webapp/webapp/models/classifier.pkl', 'rb'))
#ferti = pickle.load(open('C:/Users/parvathi143/Desktop/webapp/webapp/models/fertilizer_model.pkl', 'rb'))

#leaf_model = keras.models.load_model('models/Leaf Disease.h5')
#model_path = "C:/Users/parvathi143/Desktop/webapp/webapp/models/Leaf_disease.h5"

try:
    leaf_model = keras.models.load_model('C:/Users/parvathi143/Desktop/webapp/webapp/notebooks/notebooks/leaf_disease.h5')
    print("✅ Leaf Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Leaf Model: {e}")
    leaf_model = None  # Prevent further errors if model load fails

label_name = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 
    'Cherry Powdery mildew', 'Cherry healthy', 
    'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 
    'Corn Northern Leaf Blight', 'Corn healthy', 
    'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy',
    'Peach Bacterial spot', 'Peach healthy', 
    'Pepper bell Bacterial spot', 'Pepper bell healthy', 
    'Potato Early blight', 'Potato Late blight', 'Potato healthy', 
    'Strawberry Leaf scorch', 'Strawberry healthy',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 
    'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites', 'Tomato Target Spot', 
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]


OPENWEATHER_API_KEY = '2c097064e171dffa8c3b2882c4f92994'
OPENWEATHER_URL = 'http://api.openweathermap.org/data/2.5/weather'


def get_weather_data(city_name):
    params = {
        'q': city_name,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'  
    }
    response = requests.get(OPENWEATHER_URL, params=params)
    weather_data = response.json()
    
    if response.status_code == 200:
        temperature = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        rainfall = weather_data['rain']['1h'] if 'rain' in weather_data else 0
        return temperature, humidity, rainfall
    else:
        return None, None, None


def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    predicted_crop = recommendation_model.predict(input_features)[0]
    return predicted_crop


def predict_yield(year, season, crop, area):
    test_row = pd.read_csv('C:/Users/parvathi143/Desktop/webapp/webapp/static/x_test.csv').head(1)
    test_row['Crop_Year'] = year

    for column in test_row.columns[2:]:
        test_row[column] = 1 if season in column or crop in column else 0

   
    test_row['Area'] = area
    production = yield_model.predict(test_row)[0]
    
    yield_prediction = production / test_row['Area'].values[0]
    return production, yield_prediction
 
fertilizer_df = pd.read_csv('C:/Users/parvathi143/Desktop/webapp/webapp/Dataset_collection/f2.csv')  # Fetch from database or CSV
fertilizer_names = fertilizer_df['Fertilizer'].unique().tolist()  # Get unique fertilizer names
encode_ferti = LabelEncoder()
encode_ferti.fit(fertilizer_names)  # Fit Label Encoder with actual fertilizer names

def predict_fertilizer(temp, humi, mois, soil, crop, nitro, pota, phosp):
    #input = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]
    #predicted_index = ferti.predict(input)[0]  # Extract single value
    input_data = [[int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]]
    
    # Predict fertilizer index
    predicted_index = ferti.predict(input_data)[0]

    # Convert index to fertilizer name
    fertilizer_name = encode_ferti.inverse_transform([predicted_index])[0]
    return fertilizer_name 
    # Convert encoded value to fertilizer name
    #prediction = ferti.classes_[model.predict([input])]
    #fertilizer_name = encode_ferti.inverse_transform([prediction])#[0]
    #return prediction
    #prediction = ferti.classes_[model.predict([input])]
    # return prediction


def predict_leaf_disease(image_bytes):
    if leaf_model is None:
        return "❌ Leaf Model Not Loaded"
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
    predictions = leaf_model.predict(normalized_image)
    #return label_name[np.argmax(predictions)] if np.max(predictions) * 100 >= 80 else "Try Another Image"
    if predictions[0][np.argmax(predictions)] * 100 >= 80:
        return label_name[np.argmax(predictions)]
    else:
        return "Try Another Image"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        ph = float(request.form['ph'])
        city_name = request.form['city']  

        temperature, humidity, rainfall = get_weather_data(city_name)

        if temperature is not None:
            predicted_crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        else:
            predicted_crop = "Could not fetch weather data. Please try again."

        return render_template('recommend.html', predicted_crop=predicted_crop)

    return render_template('recommend.html', predicted_crop=None)


@app.route('/yield', methods=['GET', 'POST'])
def crop_yield():
    if request.method == 'POST':
        year = int(request.form['year'])
        season = request.form['season']
        crop = request.form['crop']
        area = float(request.form['area'])

        production,yield_prediction = predict_yield(year, season, crop, area)

        return render_template('yield.html', production=production,yield_prediction =yield_prediction )

    return render_template('yield.html')

@app.route('/fertilizer', methods=['GET','POST'])
def predict_ferti():
    if request.method == 'POST':
        temp = request.form.get('temp')
        humi = request.form.get('humid')
        mois = request.form.get('mois')
        soil = request.form.get('soil')
        crop = request.form.get('crop')
        nitro = request.form.get('nitro')
        pota = request.form.get('pota')
        phosp = request.form.get('phos')

        if None in (temp, humi, mois, soil, crop, nitro, pota, phosp) or not all(val.isdigit() for val in (temp, humi, mois, soil, crop, nitro, pota, phosp)):
            return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')
        #fertilizer_name = predict_fertilizer(temp, humi, mois, soil, crop, nitro, pota, phosp)
        try:
            # Get fertilizer prediction
            fertilizer_name = predict_fertilizer(temp, humi, mois, soil, crop, nitro, pota, phosp)
            return render_template('Model1.html', x=fertilizer_name)
        
        except Exception as e:
            return render_template('Model1.html', x=f"Error: {str(e)}")  # Show error message if any issue
    #res = predict_fertilizer(temp, humi, mois, soil, crop, nitro, pota, phosp)
    return render_template('Model1.html', x="Please submit the form to get a prediction.")



@app.route('/leaf-disease', methods=['GET', 'POST'])
def leaf_disease():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            image_bytes = uploaded_file.read()
            predicted_disease = predict_leaf_disease(image_bytes)
            image_data = base64.b64encode(image_bytes).decode('utf-8')
            return render_template('leaf_disease.html', predicted_disease=predicted_disease, image=image_data)

    return render_template('leaf_disease.html', predicted_disease=None)

if __name__ == '__main__':
    app.run(debug=True)




