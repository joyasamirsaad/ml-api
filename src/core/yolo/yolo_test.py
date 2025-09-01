from pathlib import Path
from ultralytics import YOLO

def test_model(model_name: str, data_yaml: str):
    if not model_name and not Path(f'models/{model_name}/weights/best.pt').exists():
        return {"error": "Model not found."}
    if not data_yaml or not Path(f'train_data/{data_yaml}').exists():
        return {"error": "Dataset YAML file not found."}
    
    model = YOLO(f'models/{model_name}/weights/best.pt')
    results = model.val(data=f"train_data/{data_yaml}", project="validation", name="val", exist_ok=True)
    return {"message": "Testing completed", "metrics": results.results_dict}