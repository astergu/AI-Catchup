from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# Placeholder for a pre-trained plant recognition model
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dummy plant database with recommendations
PLANT_DATABASE = {
    "fern": ["moss", "shade-loving flowers"],
    "cactus": ["succulents", "desert flowers"],
    "rose": ["lavender", "hydrangea"],
}

@app.route('/identify', methods=['POST'])
def identify_plant():
    file = request.files['image']
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
    
    predicted_label = "fern"  # Placeholder for real label mapping
    recommendations = PLANT_DATABASE.get(predicted_label, [])
    
    return jsonify({"identified_plant": predicted_label, "recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=False)
