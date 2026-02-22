 
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Define input data schema
class PatientData(BaseModel):
    age: float
    sex: float# bool
    cp: float # Chest pain type
    trestbps: float # bool # Resting blood pressure (mm Hg)
    chol: float # Serum cholesterol (mg/dl)
    fbs: float # bool # Fasting blood sugar
    restecg: float # Resting electrocardiographic results
    thalach: float # Maximum heart rate achieved
    exang: float # bool # Exercise induced angina
    oldpeak: float # ST depression induced by exercise relative to rest
    slope: float # Slope of the peak exercise ST segment
    ca: float # Number of major vessels (0–3) colored by fluoroscopy
    thal: float # Thalassemia type (normal, fixed defect, reversible defect)

    class Config:
        json_schema_extra = {
            "example": {
                "age":42,
                "sex":1,
                "cp":3,
                "trestbps":148,
                "chol":244,
                "fbs":0,
                "restecg":0,
                "thalach":178,
                "exang":0,
                "oldpeak":0.8,
                "slope":2,
                "ca":2,
                "thal":2,
            }
        }
    
# Initialise FastAPI app
app = FastAPI(
    title="Heart Disease Predictor",
    description="Predicts heart disease from physiological features",
    version="1.0.0"
)

# Load the trained model
model_path = os.path.join("models", "heart_disease_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict_progression(patients: PatientData):
    """
    Predict diabete progression score
    """

    # Convert input to numpy array
    features = np.array([[
        patients.age, patients.sex, patients.cp,patients.trestbps,
        patients.chol, patients.fbs, patients.restecg, patients.thalach,
        patients.exang, patients.oldpeak, patients.slope,patients.ca,
        patients.thal,
    ]])

    # Make prediction 
    prediction = model.predict(features)[0]

    # Return result with additional context
    return {
        "predicted_score": round(predict_progression, 2),
        "interpreation": get_interpretation(prediction)
    }

def get_interpretation(prediction):
    """
    Provide human-readable interpretable of the score
    """
    return "Test"

@app.get("/")
def health_check():
    return {"status": "healthy",
            "model": "heart_disease_v1"}

