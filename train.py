import os
from pathlib import Path
from ultralytics import YOLO
import yaml
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
def train(yaml_file: Path):
    print(f"Training model with dataset: {yaml_file}")
    # reading file to check dataset path existance
    with open(yaml_file, 'r') as file:
        prime_service = yaml.safe_load(file)
        path = prime_service.get('path', "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path {path} does not exist.")
    
    results = model.train(
        data=str(yaml_file),
        imgsz=640,
        epochs=100,
        batch=8,
        project="models",
        name='yolov8n_custom'
    )
    print("Training completed.")
    return results