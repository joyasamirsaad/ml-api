from pathlib import Path
from src.core.yolo.yolo_train import train as train_model
from src.core.yolo.yolo_inference import inference as run_inference
from src.core.yolo.yolo_test import test_model
from src.core.yolo.yolo_fine_tune import fine_tuning as fine_tune_model
from src.core.yolo.yolo_video_inference import video_inference as video_inference_model
from fastapi import UploadFile, File

def train(yaml_file: Path):
    return train_model(yaml_file)

async def inference(file: UploadFile):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')): 
        return {"error": "Invalid file type. Only PNG and JPG files are allowed."}
    
    # saving the uploaded file 
    #global file_location
    folder_location = Path("images/original")
    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f: # w: for write, b: for binary
        f.write(await file.read()) # takes bytes from read and writes to the file

    # run inference
    detection_result = run_inference(file_location)
    return {"detection_result": detection_result}

def test(model_name:str, data_yaml: str):
    return test_model(model_name, data_yaml)

def fine_tune(model: str, file: UploadFile):
    return fine_tune_model(model, file)

async def video_inference(model: str, file: UploadFile):
    return await video_inference_model(file)
