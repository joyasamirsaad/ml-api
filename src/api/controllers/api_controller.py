from fastapi import APIRouter, File, UploadFile
from src.application.services import application_service

router = APIRouter()
 
@router.post("/train/{model}")
def train(model: str):
    result = application_service.train(model)
    return {"model": model, "status": result}

@router.post("/inference/{model}")
async def inference(model: str, file: UploadFile = File(...)):
    result = await application_service.inference(model, file)
    return {"model": model, "predictions": result}

@router.post("/test/{model}")
def test_model(model: str, dataset_path: str):
    result = application_service.test(model, dataset_path)
    return {"model": model, "status": "completed", "metrics": result}

@router.post("/metrics")
async def upload_metrics(file: UploadFile = File(...)):
    result = await application_service.plot_metrics(file)
    return result

@router.post("/augmentation")
async def augmentation(file: UploadFile = File(...)):
    result = await application_service.augmentation(file)
    return result

@router.post("/fine-tune/{model}")
async def fine_tune(model: str, file: UploadFile = File(...)):
    result = await application_service.fine_tune(model, file)
    return {"model": model, "status": "fine-tuning not yet implemented"}

@router.post("/heatmap")
async def heatmap(file: UploadFile = File(...)):
    result = await application_service.heatmap(file)
    return result

@router.post("/video/{model}")
async def video_inference(model: str, file: UploadFile = File(...)):
    result = await application_service.video_inference(model, file)
    return result
 # still have video, test and fine tuning endpoints to do for effddet