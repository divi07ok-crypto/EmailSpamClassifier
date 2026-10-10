
from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "spam_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "error": "Please enter a message."
        }), 400

    prediction = int(model.predict([message])[0])

    result = "Spam" if prediction == 1 else "Ham"

    probabilities = model.predict_proba([message])[0]
    spam_probability = float(probabilities[1]) * 100

    return jsonify({
        "prediction": result,
        "spam_probability": round(spam_probability, 2)
    })


def open_browser():
    import time
    import webbrowser
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
