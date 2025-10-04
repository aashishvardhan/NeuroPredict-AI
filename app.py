import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import traceback

# --- Initialize ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Configure Chatbase API ---
try:
    CHATBASE_API_KEY = os.environ["CHATBASE_API_KEY"]
    CHATBOT_ID = os.environ["CHATBOT_ID"]
    CHATBASE_API_URL = "https://www.chatbase.co/api/v1/chat"
    HEADERS = {"Authorization": f"Bearer {CHATBASE_API_KEY}"}
    print("✅ Chatbase client configured successfully!")
except KeyError:
    print("❌ ERROR: CHATBASE_API_KEY or CHATBOT_ID environment variable not set.")
    CHATBASE_API_KEY = None

# --- Load the Stroke Prediction Model ---
MODEL_PATH = 'neuro_predict_model.h5'
stroke_model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Stroke prediction model loaded successfully!")

# --- Preprocessing Function for a Single Image ---
def preprocess_image(image):
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)

# --- API Endpoint for Stroke Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img = Image.open(file.stream)
        processed_img = preprocess_image(img)
        prediction = stroke_model.predict(processed_img)
        prediction_value = prediction[0][0]
        result = "Hemorrhage Detected" if prediction_value > 0.5 else "Normal"
        return jsonify({
            'prediction': result,
            'confidence_score': float(prediction_value)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- API Endpoint for the Chatbase AI Assistant ---
@app.route('/chatbot', methods=['POST'])
def chatbot():
    if not CHATBASE_API_KEY:
        return jsonify({'error': 'API key not configured on server'}), 500
        
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        payload = {
            "messages": [{"content": user_message, "role": "user"}],
            "chatbotId": CHATBOT_ID,
            "stream": False,
            "temperature": 0
        }
        
        response = requests.post(CHATBASE_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status() 
        
        response_data = response.json()
        generated_text = response_data.get('text', 'Sorry, I could not get a valid response.')

        return jsonify({'response': generated_text})
    except Exception as e:
        print("\n--- AN ERROR OCCURRED IN THE CHATBOT ---")
        traceback.print_exc()
        print("------------------------------------------\n")
        return jsonify({'error': 'An error occurred on the server'}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)

