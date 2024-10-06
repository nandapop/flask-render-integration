from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("/Users/nandapop/Documents/Bootcamp/flask-render-integration/models/classificador_random_forest_with_encoders.sav", "rb"))
df = pd.read_csv("/Users/nandapop/Documents/Bootcamp/flask-render-integration/data/raw/world_AQI.csv")
df_encoded = pd.read_csv("/Users/nandapop/Documents/Bootcamp/flask-render-integration/data/processed/df_enconded.csv")

classifier = model['classifier']
country_encoder = model['country_encoder']
city_encoder = model['city_encoder']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_locations', methods=['GET'])
def get_locations():
    df['Country'] = df['Country'].fillna('Unknown').astype(str)
    df['City'] = df['City'].fillna('Unknown').astype(str)

    countries_cities = {}

    for country in df['Country'].unique():
        cities = sorted(df[df['Country'] == country]['City'].unique().tolist())
        countries_cities[country] = cities
    return jsonify(countries_cities)


@app.route('/predict', methods=['POST'])
def predict():
    selected_country = request.form['country']
    selected_city = request.form['city']

    encoded_country = country_encoder.transform([selected_country])[0]
    encoded_city = city_encoder.transform([selected_city])[0]
    
    data = df_encoded[(df_encoded['Country'] == encoded_country) & 
                                (df_encoded['City'] == encoded_city)]

    if data.empty:
        return f"No data found for city {selected_city}, country {selected_country}", 404
    
    aqi_value = data['AQI Value'].values[0]
    co_aqi_value = data['CO AQI Value'].values[0]
    ozone_aqi_value = data['Ozone AQI Value'].values[0]
    no2_aqi_value = data['NO2 AQI Value'].values[0]
    pm25_aqi_value = data['PM2.5 AQI Value'].values[0]
    
    co_aqi_category = data['CO AQI Category'].values[0]
    ozone_aqi_category = data['Ozone AQI Category'].values[0]
    no2_aqi_category = data['NO2 AQI Category'].values[0]
    pm25_aqi_category = data['PM2.5 AQI Category'].values[0]

    features = np.array([[encoded_country, encoded_city, aqi_value, co_aqi_value, ozone_aqi_value, no2_aqi_value, pm25_aqi_value, co_aqi_category, ozone_aqi_category]])
    print("Features shape:", features)
    
    try:
        prediction = classifier.predict(features)
        print("Prediction:", prediction)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Prediction error: {e}", 500

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)