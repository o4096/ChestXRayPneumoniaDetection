import json
import torch
import torch.nn as nn
from torchvision import transforms, models

PATH_MODEL_CONFIG   = 'xray_classifier.json'
PATH_MODEL          = 'xray_classifier.pth'

def load_config(path:str)->dict:
	with open(path, 'r') as f:
		config= json.load(f)
	return config

def load_your_model(model_path:str=PATH_MODEL, config_path:str=PATH_MODEL_CONFIG, device=None):
    '''
    Load the saved model for inference
    
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
    '''    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(config_path)

    # Create model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    
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

