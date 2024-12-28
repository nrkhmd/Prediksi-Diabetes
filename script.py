from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import pickle
import logging
import numpy as np

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Load the model
try:
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
except Exception as e:
    logging.error(f"Error loading model: {e}")
    diabetes_model = None

app = FastAPI()

# Tambahkan CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    if diabetes_model is None:
        return {"status": "error", "message": "Model not loaded"}

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
        result = "Positif Diabetes" if prediction[0] == 1 else "Negatif Diabetes"

        return {"status": "success", "prediction": result}

    except Exception as e:
        # Tangani error jika terjadi masalah
        logging.error(f"Error during prediction: {e}")
        return {"status": "error", "message": str(e)}
