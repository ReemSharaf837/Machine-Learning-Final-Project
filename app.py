from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# Load the ORIGINAL model, scaler, and feature columns from your notebook
model = joblib.load('final_voting_model.pkl')
scaler = joblib.load('scaler.pkl') 
feature_columns = joblib.load('model_features.pkl')

app = Flask(__name__)

def convert_age_to_category(age):
    # ... (function is unchanged)
    age = float(age)
    if age < 25: return 'Age 18 to 24'
    elif age < 30: return 'Age 25 to 29'
    elif age < 35: return 'Age 30 to 34'
    elif age < 40: return 'Age 35 to 39'
    elif age < 45: return 'Age 40 to 44'
    elif age < 50: return 'Age 45 to 49'
    elif age < 55: return 'Age 50 to 54'
    elif age < 60: return 'Age 55 to 59'
    elif age < 65: return 'Age 60 to 64'
    elif age < 70: return 'Age 65 to 69'
    elif age < 75: return 'Age 70 to 74'
    elif age < 80: return 'Age 75 to 79'
    else: return 'Age 80 or older'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Collect form values into a dictionary
        form_data = request.form.to_dict()
        
        # --- DIAGNOSTIC 1: Print the raw data received from the form ---
        print("\n--- 1. Raw Form Data ---")
        print(form_data)
        
        # Create a single-row DataFrame with all model columns, initialized to 0
        input_df = pd.DataFrame(columns=feature_columns)
        input_df.loc[0] = 0

        # Step 2: Fill the DataFrame with user data BEFORE scaling
        input_df['BMI'] = float(form_data.get('BMI'))
        input_df['SleepHours'] = float(form_data.get('SleepHours'))
        input_df['PhysicalHealthDays'] = float(form_data.get('PhysicalHealthDays'))
        
        if 'MentalHealthDays' in feature_columns:
            input_df['MentalHealthDays'] = 0 
        
        # Step 3: Scale numeric features
        numeric_cols_to_scale = [col for col in scaler.feature_names_in_ if col in input_df.columns]
        input_df[numeric_cols_to_scale] = scaler.transform(input_df[numeric_cols_to_scale])

        # Step 4: Set the one-hot encoded flags
        age_cat = convert_age_to_category(float(form_data.get('Age')))
        
        # Build a list of one-hot columns to set to 1
        active_features = {
            f"Sex_{form_data.get('Sex')}",
            f"SmokerStatus_{form_data.get('SmokerStatus')}",
            f"GeneralHealth_{form_data.get('GeneralHealth')}",
            f"AgeCategory_{age_cat}",
        }
        
        if form_data.get('HadStroke') == 'Yes': active_features.add('HadStroke_Yes')
        if form_data.get('HadAngina') == 'Yes': active_features.add('HadAngina_Yes')
        if form_data.get('HadDiabetes') == 'Yes': active_features.add('HadDiabetes_Yes')
        if form_data.get('DifficultyWalking') == 'Yes': active_features.add('DifficultyWalking_Yes')
        
        # --- DIAGNOSTIC 2: Print the features we are about to set to 1 ---
        print("\n--- 2. Features to be set to 1 ---")
        print(active_features)

        for col in active_features:
            if col in input_df.columns:
                input_df[col] = 1

        # --- DIAGNOSTIC 3: Print the final probability value ---
        proba = model.predict_proba(input_df[feature_columns])[0][1]
        print(f"\n--- 3. Prediction Probability for High Risk: {proba:.4f} ---")

        # Step 5: Predict
        prediction = 1 if proba > 0.25 else 0
        result = "💔 High Risk of Heart Attack" if prediction == 1 else "💚 Low Risk of Heart Attack"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        error_message = f'❌ Error during prediction: {str(e)}'
        print(error_message)
        return render_template('index.html', prediction_text=error_message)

if __name__ == '__main__':
    app.run(debug=True)