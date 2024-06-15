from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

app = Flask(__name__)
data = pd.read_csv('data.csv')

# Load the model
model = joblib.load('LR_model.pkl')

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    cities = sorted(data['city'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, cities=cities)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #data = request.get_json()
    #bedrooms = data['bedrooms']
    #bathrooms = data['bathrooms']
    #city = data['city']
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    city = request.form.get('city')
    # Prepare the input for the model
    #input_data = np.array([[bedrooms, bathrooms, city]])
     # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, city]],
                               columns=['beds', 'baths', 'city'])
    
    # Make prediction
    predicted_price = model.predict(input_data)[0]
    
    return (predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
