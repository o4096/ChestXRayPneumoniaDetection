 # app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch, io
from utils import load_your_model
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


app = FastAPI(title='Pneumonia Detection API')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], 
    allow_methods=['*'],
    allow_headers=['*'],
)

model, config, preprocess = load_your_model()

@app.get("/")
def root():
    return FileResponse("index.html")


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        classification = config['classes'][probs.argmax()]
        confidence = float(probs.max())
        pct = round(confidence * 100, 1)
        risk_level = ("HIGH" if pct > 61 else "MEDIUM" if pct > 25 else "LOW") if classification == 'PNEUMONIA' else ("LOW" if pct > 61 else "MEDIUM" if pct > 25 else "HIGH")
    return {'class': classification,
            'confidence': confidence,
            'risk_level': risk_level}
           

