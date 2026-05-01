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

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = [[
        float(data["study_hours"]),
        float(data["sleep_hours"]),
        float(data["attendance"]),
        float(data["internet_usage"]),
        float(data["stress_level"])
    ]]

    prediction, probability = predict_student(features)

    result = "PASS" if prediction == 1 else "FAIL"

    return jsonify({
        "prediction": result,
        "probability": round(probability * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
