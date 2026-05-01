from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory
from backend.Model.predict import predict_student
import os

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Serve frontend
@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({
        "message": "Causal Representation Learning Backend Running"
    })

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
        "probability": round(probability * 100, 2)
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

