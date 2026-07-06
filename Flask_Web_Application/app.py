import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join("..", "Saved_Models", "XGBoost_Model.pkl")
SCALER_PATH = os.path.join("..", "Saved_Models", "Preprocessor_Scaler.pkl")

def get_ml_assets():
    with open(MODEL_PATH, "rb") as m:
        model = pickle.load(m)
    with open(SCALER_PATH, "rb") as s:
        scaler = pickle.load(s)
    return model, scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        annual_rf = float(request.form['annual_rainfall'])
        visibility = float(request.form['cloud_visibility'])
        seasonal_rf = float(request.form['seasonal_rainfall'])
        
        model, scaler = get_ml_assets()
        raw_features = np.array([[annual_rf, visibility, seasonal_rf]])
        scaled_features = scaler.transform(raw_features)
        
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] * 100
        
        result_status = "HIGH RISK" if prediction == 1 else "LOW RISK"
        
        return render_template('result.html', 
                               result=result_status, 
                               probability=round(probability, 2),
                               annual_rf=annual_rf, 
                               visibility=visibility, 
                               seasonal_rf=seasonal_rf)
    except Exception as e:
        return f"Backend Error: Ensure you trained your model first. Detail: {str(e)}"

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
