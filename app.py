from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import io
import os
import requests
from io import BytesIO
import pickle
from ultralytics import YOLO

app = Flask(__name__)

# === Path & Config ===
LABEL_ENCODER_PATH = "label_encoder.pkl"

# === Load Label Encoder ===
label_encoder = None
if os.path.exists(LABEL_ENCODER_PATH):
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

# === Model Residual untuk Klasifikasi Penyakit ===
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return torch.relu(self.conv(x) + self.skip(x))

class MiniResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = ResidualBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.layer3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.layer4 = ResidualBlock(128, 128)
        self.pool4 = nn.MaxPool2d(2)
        self.layer5 = ResidualBlock(128, 128)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
        x = self.pool5(self.layer5(x))
        x = self.classifier(x)
        return x

# === Load Models ===  
def load_model(path, is_cnn=True):
    if is_cnn:
        loaded = torch.load(path, map_location=torch.device('cpu'))
        model = MiniResNet(num_classes=len(label_encoder.classes_))
        model.load_state_dict(loaded['state_dict'] if 'state_dict' in loaded else loaded)
        model.eval()
        return model
    else:
        return YOLO(path)


model_cnn = load_model("model_trained.pt", is_cnn=True)
model_yolo = load_model("best_model6.pt", is_cnn=False)

# === Transforms ===
transform_common = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
transform_cnn = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === CNN Prediction ===
def predict_cnn(image):
    image = image.convert('RGB')
    input_tensor = transform_cnn(image).unsqueeze(0)
    with torch.no_grad():
        output = model_cnn(input_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = int(predicted.item())
        label = label_encoder.inverse_transform([class_idx])[0] if label_encoder else str(class_idx)
        return label

# === YOLO Detection ===
def predict_yolo(image):
    import numpy as np
    image = image.convert('RGB')
    img_resized = transform_common(image)
    results = model_yolo(img_resized.unsqueeze(0))[0]

    draw = ImageDraw.Draw(image)
    if results.boxes is not None:
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"Class {int(cls)}: {conf:.2f}", fill="red")

    
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io

# === UTILS ===
def load_image_from_request():
    file = request.files.get('image')
    if not file:
        raise ValueError("No image uploaded")
    return Image.open(io.BytesIO(file.read()))

def load_image_from_url():
    data = request.get_json()
    url = data.get('image_url')
    if not url:
        raise ValueError("No image_url provided")
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

# === CNN Endpoint ===
@app.route('/predict-disease/upload', methods=['POST'])
def predict_disease_upload():
    try:
        image = load_image_from_request()
        result = predict_cnn(image)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-disease/url', methods=['POST'])
def predict_disease_url():
    try:
        image = load_image_from_url()
        result = predict_cnn(image)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === YOLO Detection Endpoint ===
@app.route('/detect-weed/upload', methods=['POST'])
def detect_weed_upload():
    try:
        image = load_image_from_request()
        result_image = predict_yolo(image)
        return send_file(result_image, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect-weed/url', methods=['POST'])
def detect_weed_url():
    try:
        image = load_image_from_url()
        result_image = predict_yolo(image)
        return send_file(result_image, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run Flask App ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
