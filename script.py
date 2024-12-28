from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import logging
import numpy as np

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Load the model
scaler, diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

app = FastAPI()

# Prediction route
@app.post('/api/predict')
async def predict(
    Pregnancies: float = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: float = Form(...),
):
    try:
        # Prepare the input for the model
        input_data = np.array([[
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]])

        # Log input data
        logging.info(f"Input data: {input_data}")

        # Make prediction
        prediction = diabetes_model.predict(input_data)

        # Log hasil prediksi
        logging.info(f"Model prediction: {prediction}")

        # Logika prediksi
        if prediction[0] == 1:
            result = "Positif Diabetes"  # Jika prediksi 1
        elif prediction[0] == 0:
            result = "Negatif Diabetes"  # Jika prediksi 0
        else:
            result = "Prediksi tidak valid"  # Jika model mengembalikan nilai selain 0 atau 1

        return {"prediction": result}

    except Exception as e:
        # Tangani error jika terjadi masalah
        logging.error(f"Error during prediction: {e}")
        return {"error": str(e)}
