 # app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch, io
import torchvision.transforms as transforms
from torchvision import transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn
from PIL import Image
import json




def load_your_model(model_path, config_path, device='cpu'):
    """
    Load the saved MobileNetV2 model for inference
    
    Args:
        model_path: Path to the .pth file
        config_path: Path to the config JSON file
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Loaded model
        config: Configuration dict
        transform: Transform pipeline
    
    Example Usage:
        model, config, transform = load_saved_model(model_path, config_path)
    """    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = mobilenet_v2(weights=None)
    model.classifier[-1] = nn.Linear(model.last_channel, config['num_classes'])
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((config['input_size'], config['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    return model, config, transform

def preprocess(img: Image.Image):
    return transform(img)

app = FastAPI(title="Pneumonia Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# model = load_your_model("mobilenetv2_facemask.pth")  # Load trained model
model, config, transform = load_your_model("best_model.pth", "model_config.json")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
    width: int = Form(0),
    height: int = Form(0),
):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

    if width > 0 and height > 0:
        left = max(0, x)
        upper = max(0, y)
        right = min(img.width, x + width)
        lower = min(img.height, y + height)
        if right > left and lower > upper:
            img = img.crop((left, upper, right, lower))

    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return {"class": config['class_names'][probs.argmax()],
            "confidence": float(probs.max())}