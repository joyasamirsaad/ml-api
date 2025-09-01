import json
from pathlib import Path
from fastapi import UploadFile
from moviepy import VideoFileClip
from ultralytics import YOLO


async def video_inference(file: UploadFile): 
    if not file.filename.endswith(('.mp4', '.mjpeg')): 
        return {"error": "Invalid file type. Only mp4 and mjepg files are allowed."}
    
    # saving the uploaded file 
    folder_location = Path("videos/original")
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
    results = model.track(video_path, stream=True, tracker="botsort.yaml", save=True, save_frames=True, project="videos", name="detect", exist_ok=True) # detecting
    # tracker to track the same object accross frames and not give a new id

    # saving video data frame by frame
    objects = [] 
    labels = []
    framenb = 0
    c = 0
    for r in results:
        #r.save(filename=f"detection_{video_name}")
        timestamp = framenb / fps
        for box in r.boxes:
            if box is None: 
                continue
            if box.id is None: 
                box_id = -c
                c += 1
            else:
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
    json_path = Path("videos") / "detect" / f"{video_name_without_ext}.json" # save json in the same directory as the video
    
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
    summary_path = Path("videos") / "detect" / f"{video_name_without_ext}_summary.json"
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