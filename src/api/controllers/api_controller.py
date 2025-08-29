from fastapi import APIRouter
from application.services import application_service

router = APIRouter()

@router.get("/ping")
def ping():
    return {"message": "pong"}

@router.get("/compute/{x}")
def compute(x: int):
    result = application_service.square_number(x)
    return {"input": x, "result": result}
 
@router.post("/train/{model}")
def train(model: str):
    result = application_service.train(model)
    return {"model": model, "status": result}

@router.post("/inference/{model}")
async def inference(model: str, image_path: str):
    result = application_service.inference(model, image_path)
    return {"model": model, "predictions": result}

@router.post("/fine-tune/{model}")
def fine_tune(model: str):
    result = application_service.fine_tune(model)
    return {"model": model, "status": result}   
