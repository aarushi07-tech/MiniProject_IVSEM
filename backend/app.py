import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from flask import Flask, request, jsonify
from Model.predict import predict_student

app = Flask(__name__)

@app.route("/")
def home():
    return {
        "message":"Causal Representation Learning Backend Running"
    }

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    study_hours = float(data["study_hours"])
    sleep_hours = float(data["sleep_hours"])
    attendance = float(data["attendance"])
    internet_usage = float(data["internet_usage"])
    stress_level = float(data["stress_level"])

    features = [[
        study_hours,
        sleep_hours,
        attendance,
        internet_usage,
        stress_level
    ]]

    prediction, probability = predict_student(features)

    result = "PASS" if prediction == 1 else "FAIL"

    response = {
        "prediction": result,
        "probability": round(probability*100,2)
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
