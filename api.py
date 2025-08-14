from pathlib import Path
from fastapi import FastAPI , UploadFile, File
from ultralytics import YOLO
from moviepy import VideoFileClip

# creating instance of the FastAPI application
app = FastAPI()

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

    detection_result = {
        "image_path": str(f"detection_{image_name}"),
        "objects_detected": objects
    }

    return { "detection_result": detection_result }

# POST Request video
@app.post("/video") # api endpoint
async def upload_file(file: UploadFile = File(...)): 
    if not file.filename.endswith(('.mp4', '.mjpeg')): 
        return {"error": "Invalid file type. Only mp4 and mjepg files are allowed."}
    
    # saving the uploaded file 
    folder_location = Path("videos")
    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f: # w: for write, b: for binary
        f.write(await file.read()) # takes bytes from read and writes to the file

    # frames per second for timestamp
    clip = VideoFileClip(str(file_location))
    fps = clip.fps  
    clip.close()

    # function to detect
    detection_result = await vid_detection(file_location, file.filename, fps)

    # return the labeled image path and the objects detected
    return detection_result

async def vid_detection(video_path: Path, video_name: str, fps: int):
    # model
    model = YOLO("yolov8n.pt") # trained with coco8.yaml dataset
    results = model.track(video_path, stream=True, tracker="botsort.yaml", save=True, project=".", name="detect", exist_ok=True) # detecting
    # tracker to track the same object accross frames and not give a new id

    # saving video frame by frame
    objects = [] 
    framenb = 0
    for r in results:
        #r.save(filename=f"detection_{video_name}")
        timestamp = framenb / fps
        for box in r.boxes:
            idx = int(box.cls)       
            conf = float(box.conf)   
            bbox = box.xyxy.tolist()[0]
            label = model.names[idx]
            objects.append({
                "label": label,
                "frame": framenb,
                "timestamp": timestamp,
                "confidence": round(conf, 2),
                "bbox": bbox
            })

        framenb += 1

    detection_result = {
        "video_path": str(Path(".") / "detect" / f"{video_name}"),
        "objects_detected": objects
    }

    return { "detection_result": detection_result }