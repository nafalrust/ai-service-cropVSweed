from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import requests
from io import BytesIO
import pickle

# === Konfigurasi ===
LABEL_ENCODER_PATH = "label_encoder.pkl"

app = Flask(__name__)

# === Arsitektur Model Langsung di Sini ===
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 28x28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 kelas: crop vs weed
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# === Load Label Encoder ===
label_encoder = None
if os.path.exists(LABEL_ENCODER_PATH):
    try:
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        print(f"Error loading label encoder: {e}")
else:
    print(f"Label encoder file '{LABEL_ENCODER_PATH}' not found.")

# === Load Kedua Model ===
def load_model(path):
    try:
        loaded = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(loaded, dict) and 'state_dict' in loaded:
            model = MyModel()
            model.load_state_dict(loaded['state_dict'])
        else:
            model = loaded
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        return None

model_entire = load_model("model_entire.pt")
model_best = load_model("best_model6.pt")

# === Preprocessing ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Fungsi prediksi umum ===
def run_prediction(image, model):
    img = image.convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        _, predicted = torch.max(output, 1)
        result = int(predicted.item())
        label = label_encoder.inverse_transform([result])[0] if label_encoder else str(result)
    return label

# === Endpoint Upload Gambar ===
@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.args.get('model', 'entire')
    model = model_entire if model_name == 'entire' else model_best

    if model is None:
        return jsonify({'error': f'Model "{model_name}" not loaded'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read()))
        label = run_prediction(img, model)
        return jsonify({'prediction': label, 'model_used': model_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Endpoint URL Gambar ===
@app.route('/predict_url', methods=['POST'])
def predict_url():
    model_name = request.args.get('model', 'entire')
    model = model_entire if model_name == 'entire' else model_best

    if model is None:
        return jsonify({'error': f'Model "{model_name}" not loaded'}), 500

    data = request.get_json()
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({'error': 'No image_url provided'}), 400

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        label = run_prediction(img, model)
        return jsonify({'prediction': label, 'model_used': model_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run Flask App ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
