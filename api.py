from pathlib import Path
import cv2
from fastapi import FastAPI , UploadFile, File
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
from moviepy import VideoFileClip
import json

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
    results = model.track(video_path, stream=True, tracker="botsort.yaml", save=True, save_frames=True, project=".", name="detect", exist_ok=True) # detecting
    # tracker to track the same object accross frames and not give a new id

    # saving video data frame by frame
    objects = [] 
    labels = []
    framenb = 0
    for r in results:
        #r.save(filename=f"detection_{video_name}")
        timestamp = framenb / fps
        for box in r.boxes:
            if box is None or  box.id is None: continue
            box_id = int(box.id)  # get the unique ID of the object
            idx = int(box.cls)       
            conf = float(box.conf)   
            bbox = box.xyxy.tolist()[0]
            label = model.names[idx]
            if not any(l["label"] == label and l["id"] == box_id for l in labels):
                labels.append({
                    "label": label,
                    "id": box_id
                })
            objects.append({
                "label": label,
                "frame": framenb,
                "timestamp": timestamp,
                "confidence": round(conf, 2),
                "bbox": bbox
            })

        framenb += 1

    video_name_without_ext = video_name.split(".")[0] # remove file extension
    json_path = Path(".") / "detect" / f"{video_name_without_ext}.json" # save json in the same directory as the video
    
    # save the objects detected in a json file
    objects_detected = {
        "objects_detected": objects,
    }
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the directory exists
        with open(json_path, "w") as f: # writing to json file
            json.dump(objects_detected, f, indent=4)
    except Exception as e:
        print(f"Error saving JSON: {e}")

    # summary of the detection
    summary = {
        "total_frames": framenb,  
        "total_objects": len(objects),
        "unique_objects": len(labels),
        "object_counts": {lbl: sum(1 for x in labels if x["label"] == lbl) for lbl in {l["label"] for l in labels}}
    } # lbl: unique labels { no duplication }, x: labels with the same label (lbl) 

    # saving the summary to a json file
    summary_path = Path(".") / "detect" / f"{video_name_without_ext}_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4) 
    except Exception as e:
        print(f"Error saving summary JSON: {e}")

    # "objects_detected": objects - too many objects, so saving to json
    detection_result = {
        "video_path": str(Path(".") / "detect" / f"{video_name}"),
        "json_path": str(json_path),
        "summary_path": str(summary_path),
    }

    return { "detection_result": detection_result }

@app.post("/heatmap") # api endpoint
async def heatmap(file: UploadFile = File(...)):
    if not file.filename.endswith(('.json')):
        return {"error": "Invalid file type. Only JSON files are allowed."}
  
    # reading the data from uploaded file
    contents = await file.read()          
    data = json.loads(contents.decode())

    video_name = file.filename.split(".")[0]  
    # function to create heatmap
    overlay = heatmap_image(data, video_name)

    # saving the heatmap overlay image
    heatmaps = Path("heatmaps")
    heatmaps.mkdir(parents=True, exist_ok=True)
    file_path = heatmaps / f"{video_name}_heatmap_overlay.jpg"
    cv2.imwrite(str(file_path), overlay)
    return {
        "message": "Heatmap saved successfully", 
        "filename": video_name + "_heatmap_overlay.jpg",
    }

def heatmap_image(data, video_name: str):
    frame_x = cv2.imread(f"detect/{video_name}_frames/1.jpg")
    video_height, video_width = frame_x.shape[:2] 
    # 2D array for heatmap, initializing heatmap array with zeros
    heatmap_data = np.zeros((video_height, video_width), dtype=np.float32)
    
    for obj in data['objects_detected']:
        if obj['label'] == 'person':
            x_min, y_min, x_max, y_max = map(int, obj["bbox"]) # xyxy coordinates from bounding box
            # ensure coordinates are within bounds
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(video_width-1, x_max), min(video_height-1, y_max)
            # increment heatmap in bbox region
            heatmap_data[y_min:y_max+1, x_min:x_max+1] += 1
    
    # using cv2 to create a heatmap overlay
    # normalize heatmap
    heatmap_normalized = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_normalized = heatmap_normalized.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_HOT)

    # resizing the frame to the same dimensions as the video before overlaying
    frame_x = cv2.resize(frame_x, (video_width, video_height))

    alpha = 0.6  # heatmap weight / opacity = 60%
    beta = 1 - alpha  # base frame weight / opacity = 40%

    overlay = cv2.addWeighted(frame_x, beta, heatmap_color, alpha, 0)

    return overlay