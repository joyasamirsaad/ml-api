from pathlib import Path
from fastapi import UploadFile
from ultralytics import YOLO
import yaml


async def fine_tuning(model_name: str, file: UploadFile):
    if not file.filename.endswith(('.yaml')):
        return {"error": "Invalid file type. Only YAML files are allowed."}
    if not model_name and not Path(f'models/{model_name}/weights/best.pt').exists():
        return {"error": "Model not found."}
    
    # saving the uploaded file 
    folder_location = Path("train_data")
    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f: # w: for write, b: for binary
        f.write(await file.read())
    
    # function to tune the model
    model = YOLO(f'models/{model_name}/weights/best.pt')
    search_space = {
        "lr0": (1e-5, 1e-2),  # learning rate
        "lrf": (0.1, 0.9),   # final learning rate
    }
    results = model.tune(data=f"train_data/{file.filename}", epochs=5, iterations=30, space=search_space, val=True, project="tuning", name="exp", resume=True)
    
    with open("tuning/tune/best_hyperparameters.yaml") as f:
        best_hyperparams = yaml.safe_load(f)
    return {"message": "Fine-tuning completed", "best_hyperparameters": best_hyperparams}
