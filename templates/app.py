```python
from flask import Flask, render_template, request, jsonify
import joblib
import os
import threading
import webbrowser
import time

app = Flask(__name__)

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")

model = joblib.load(MODEL_PATH)

print("Spam detection model loaded successfully!")
print("Model path:", MODEL_PATH)


@app.route("/")
def home():
    # Temporary marker to confirm the latest app.py is running
    print("DEBUG: Latest home route loaded")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "error": "Please enter a message."
        }), 400

    try:
        prediction = int(model.predict([message])[0])
        probabilities = model.predict_proba([message])[0]

        # Dataset labels: 1 = Spam, 0 = Ham
        result = "Spam" if prediction == 1 else "Ham"

        spam_probability = float(probabilities[1]) * 100

        print(
            "DEBUG:",
            repr(result),
            "| Prediction:",
            prediction,
            "| Spam probability:",
            round(spam_probability, 2)
        )

        return jsonify({
            "prediction": result,
            "spam_probability": round(spam_probability, 2)
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Unable to classify this message."
        }), 500


def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("Starting Email Spam Classifier...")
    print("Open: http://127.0.0.1:5000")

    threading.Thread(
        target=open_browser,
        daemon=True
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
