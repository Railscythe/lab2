from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
print("Flask version :" , flask.__version__)
# Load the trained model and scaler
with open('rf_reg.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Min and max price values used during training
min_price = 9.13461632544666  # Extracted from laptop_price.py
max_price = 12.69144112852859

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse numerical inputs
        data = request.form
        ram = float(data['ram'])
        weight = float(data['weight'])
        ppi = float(data['ppi'])
        hdd = float(data['hdd'])
        ssd = float(data['ssd'])

        # Parse categorical inputs
        company = data['company']
        type_name = data['type_name']
        cpu_brand = data['cpu_brand']
        gpu_brand = data['gpu_brand']
        os = data['os']

        # Prepare input data as a dictionary
        input_data = {
            'Ram': ram,
            'Weight': weight,
            'Ppi': ppi,
            'HDD': hdd,
            'SSD': ssd,
            f'Company_{company}': 1,
            f'TypeName_{type_name}': 1,
            f'Cpu_brand_{cpu_brand}': 1,
            f'Gpu_brand_{gpu_brand}': 1,
            f'Os_{os}': 1
        }

        # Convert input_data into a DataFrame and align with model features
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Scale numerical features
        numerical_cols = ['Ram', 'Weight', 'Ppi', 'HDD', 'SSD']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Predict using the model
        raw_prediction = model.predict(input_df)

        # Apply scaling logic from laptop_price.py
        y_scaled = (raw_prediction - min_price) / (max_price - min_price)
        y_scaled = y_scaled * 100000

        # Debug: Print intermediate values
        print("Raw Prediction (Price):", raw_prediction)
        print("Scaled Prediction:", y_scaled)

        return render_template('result.html', price=round(float(y_scaled[0]), 2))

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
