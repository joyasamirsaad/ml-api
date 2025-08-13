from pathlib import Path
from fastapi import FastAPI , UploadFile, File
from ultralytics import YOLO

# creating instance of the FastAPI application
app = FastAPI()

#file_location: Path = None

# POST Request
@app.post("/frame") # api endpoint
async def upload_file(file: UploadFile = File(...)): 
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')): 
        return {"error": "Invalid file type. Only PNG and JPG files are allowed."}
    
    # saving the uploaded file 
    #global file_location
    file_location = Path(file.filename)
    with open(file_location, "wb") as f: # w: for write, b: for binary
        f.write(await file.read()) # takes bytes from read and writes to the file

    # function to detect
    detection_result = await detection(file_location, file.filename)

    # return the labeled image path and the objects detected
    return detection_result

async def detection(image_path: Path, image_name: str):
    # model
    model = YOLO("yolov8n.pt") # trained with coco8.yaml dataset
    results = model(image_path) # detecting; if save=True -> saved in runs/predict
    results[0].show() # showing the new image
    results[0].save(filename=f"detection_{image_name}") # saving the new image
    
    #objects = results[0].to_json()
    objects = [] 
    for box in results[0].boxes:
        idx = int(box.cls)       
        conf = float(box.conf)   
        bbox = box.xyxy.tolist()[0]
        label = model.names[idx]
        objects.append({
            "label": label,
            "confidence": round(conf, 2),
            "bbox": bbox
        })

    global detection_result
    detection_result = {
        "image_path": str(f"{image_name}_detection"),
        "objects_detected": objects
    }

    return { "detection_result": detection_result }

# GET Request
@app.get("/frame") # api enpoint
async def display():
    global detection_result
    return detection_result
""" @app.get("/frame") # api endpoint
async def read_root():
    global file_location
    
    if not file_location.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(file_location)  """