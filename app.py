from flask import Flask, render_template, request
import joblib
import webbrowser
from threading import Timer

app = Flask(__name__)

# Load ML Model
model = joblib.load("crop_model.pkl")

# Crop Information
crop_info = {
    "rice": "High water requirement. Best grown in Kharif season.",
    "maize": "Moderate water requirement. Rich in nutrients.",
    "cotton": "Requires warm climate and moderate rainfall.",
    "coffee": "Grows well in shaded regions with good rainfall.",
    "banana": "Needs high humidity and water supply.",
    "mango": "Tropical fruit crop with high market demand."
}

# Fertilizer Recommendation
fertilizer_info = {
    "rice": "Urea, DAP, Potash",
    "maize": "Nitrogen Fertilizer",
    "cotton": "NPK Fertilizer",
    "banana": "Organic Compost",
    "mango": "Farm Yard Manure"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temp = float(request.form["temp"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    result = model.predict([[N, P, K, temp, humidity, ph, rainfall]])

    crop = result[0]

    info = crop_info.get(
        crop,
        "Suitable crop for given environmental conditions."
    )

    fertilizer = fertilizer_info.get(
        crop,
        "General NPK Fertilizer"
    )

    return render_template(
        "index.html",
        prediction=crop,
        crop_info=info,
        fertilizer=fertilizer
    )

if __name__ == "__main__":
    Timer(
        2,
        lambda: webbrowser.open("http://127.0.0.1:5000")
    ).start()

    app.run(debug=True)