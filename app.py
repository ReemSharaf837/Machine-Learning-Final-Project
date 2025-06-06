from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load('final_voting_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('model_features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form.get(f)) for f in features]
    df = pd.DataFrame([values], columns=features)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    return render_template('index.html', prediction_text=f'Prediction: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
