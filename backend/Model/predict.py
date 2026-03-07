import joblib
import numpy as np

model = joblib.load("Model/model.pkl")
scaler = joblib.load("Model/scaler.pkl")

def predict_student(input_data):

    input_array = np.array(input_data)

    scaled_data = scaler.transform(input_array)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    return prediction, probability
