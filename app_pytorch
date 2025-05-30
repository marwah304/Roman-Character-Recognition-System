import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from flask import jsonify

# App setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model definition (must match training)
class RomanCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(RomanCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load model
model_path = "roman_model.pth"
model = RomanCNN(num_classes=10)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Class labels
class_names = ["I", "II", "III", "IV", "IX", "V", "VI", "VII", "VIII",  "X"]

# Image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route("/predict", methods=["POST"])
def predict_ajax():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]

    return jsonify({"prediction": prediction})


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = Image.open(filepath).convert('RGB')
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                prediction = class_names[predicted.item()]
            image_path = filepath

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
