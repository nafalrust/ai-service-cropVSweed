from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import requests
from io import BytesIO

# Tambahkan import YOLO jika model dari Ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__)

MODEL_PATH = "best_model6.pt"  # ganti dengan path modelmu

model = None
if os.path.exists(MODEL_PATH):
    try:
        # Coba load sebagai model Ultralytics YOLO
        if YOLO is not None:
            model = YOLO(MODEL_PATH)
        else:
            # Fallback: load sebagai PyTorch model biasa
            loaded = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
            if isinstance(loaded, dict):
                # Anda harus mendefinisikan arsitektur model di sini, contoh:
                # from my_model import MyModel
                # model = MyModel()
                # model.load_state_dict(loaded['state_dict'])
                # model.eval()
                raise RuntimeError("Model file berisi state_dict. Silakan definisikan arsitektur model dan load state_dict.")
            else:
                model = loaded
                model.eval()
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")

# Preprocessing sesuai kebutuhan model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # sesuaikan dengan input modelmu
    transforms.ToTensor(),
    # transforms.Normalize([...])  # tambahkan jika model butuh normalisasi
])

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': f"Model file '{MODEL_PATH}' not found. Please make sure the file exists in the correct directory."}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)  # batch size 1

    with torch.no_grad():
        output = model(input_tensor)
        # Jika output logits, ambil argmax
        _, predicted = torch.max(output, 1)
        result = int(predicted.item())

    return jsonify({'prediction': result})

@app.route('/predict_url', methods=['POST'])
def predict_url():
    data = request.get_json()
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({'error': 'No image_url provided'}), 400

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0)  # batch size 1

        with torch.no_grad():
            output = model(input_tensor)
            # Jika output logits, ambil argmax
            _, predicted = torch.max(output, 1)
            result = int(predicted.item())

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
