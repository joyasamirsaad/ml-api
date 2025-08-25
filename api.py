from pathlib import Path
import cv2
from fastapi import FastAPI , UploadFile, File, BackgroundTasks
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO
from moviepy import VideoFileClip
import json

import yaml
from train import train
from augmentation import albumentations_transform

# creating instance of the FastAPI application
app = FastAPI()

# POST Request
@app.post("/frame") # api endpoint
async def upload_file(file: UploadFile = File(...)): 
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')): 
        return {"error": "Invalid file type. Only PNG and JPG files are allowed."}
    
    # saving the uploaded file 
    #global file_location
    folder_location = Path("images/original")
    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f: # w: for write, b: for binary
        f.write(await file.read()) # takes bytes from read and writes to the file

    # function to detect
    detection_result = await detection(file_location, file.filename)

    # return the labeled image path and the objects detected
    return detection_result

async def detection(image_path: Path, image_name: str):
    # model
    # model = YOLO("yolov8n.pt") # trained with coco8.yaml dataset
    # results = model(image_path) # detecting; if save=True -> saved in runs/predict
    model = YOLO("models/yolov8n_custom/weights/best.pt") # trained with custom dataset
    results = model.predict(source = image_path, save=True, project="images", name="detect", exist_ok=True) # detecting; if save=True -> saved in runs/predict
    results[0].show() # showing the new image
    # results[0].save(filename=f"detection_{image_name}") # saving the new image
    
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
    #frame_x = cv2.imread(f"detect/{video_name}_frames/1.jpg")
    frame_x = cv2.imread(str(sorted(Path(f"detect/{video_name}_frames").glob("*.jpg"))[0]))  # reading the first frame of the video
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
            print("BBox size:", x_max - x_min, y_max - y_min)
    
    # using cv2 to create a heatmap overlay
    # normalize heatmap
    heatmap_normalized = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_normalized = np.clip(heatmap_normalized, 0, 255).astype(np.uint8)
    
    print("Heatmap data", heatmap_data)
    print("Heatmap normalized", heatmap_normalized)
    # cv2.imshow("Heatmap Data", heatmap_normalized)  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()

    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_HOT)
    # cv2.imshow("Heatmap Color", heatmap_color)  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()

    # resizing the frame to the same dimensions as the video before overlaying
    frame_x = cv2.resize(frame_x, (video_width, video_height))

    alpha = 0.6  # heatmap weight / opacity = 60%
    beta = 1 - alpha  # base frame weight / opacity = 40%

    overlay = cv2.addWeighted(frame_x, beta, heatmap_color, alpha, 0)

    return overlay

@app.post("/train") # api endpoint
async def train_model(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if not file.filename.endswith(('.yaml', '.yml')):
        return {"error": "Invalid file type. Only YAML files are allowed."}
    
    # saving the uploaded file 
    folder_location = Path("train_data")
    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f: # w: for write, b: for binary
        f.write(await file.read()) # takes bytes from read and writes to the file
    
    # function to train the model
    background_tasks.add_task(train, file_location)
    # results = train(file_location)

@app.get("/test")
def testing(model_name: str, data_yaml: str):
    if not model_name and not Path(f'models/{model_name}/weights/best.pt').exists():
        return {"error": "Model not found."}
    if not data_yaml or not Path(f'train_data/{data_yaml}').exists():
        return {"error": "Dataset YAML file not found."}
    
    model = YOLO(f'models/{model_name}/weights/best.pt')
    results = model.val(data=f"train_data/{data_yaml}", project="validation", name="val", exist_ok=True)
    return {"message": "Testing completed", "metrics": results.results_dict}

@app.post("/augmentation") # api endpoint
async def augmentation(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        return {"error": "Invalid file type. Only ZIP files are allowed."}
    
    # saving the uploaded file 
    folder_location = Path(f"{file.filename.replace('.zip', '')}_dataset")
    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f:
        f.write(await(file.read()))
    
    # extracting the uploaded file
    zip_folder_location = Path("dataset_augmented") / file.filename.replace(".zip", "") # unique folder per ZIP, so no overwrite/PermissionError.
    zip_folder_location.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(file_location, 'r') as zip_ref:
        zip_ref.extractall(zip_folder_location)

    # function to perform augmentation
    zip_folder_location = Path(f"{zip_folder_location}/dataset/train/images")
    for img_path in zip_folder_location.rglob("*"):
        #print("for loop")
        albumentations_transform(img_path)

    return {"message": "Augmentation completed", "folder": str(zip_folder_location)}
    
@app.post("/metrics") # api endpoint
async def metrics(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv')):
        return {"error": "Invalid file type. Only CSV files are allowed."}
    
    # saving the uploaded file
    counter = 1
    c = 1
    if "tune" in file.filename:
        base_folder = Path(f"metrics/tuning{c}")
        folder_location = base_folder
        if folder_location.exists():
            c += 1
            folder_location = Path(f"metrics/tuning{c}")
    else:
        base_folder = Path(f"metrics/plots{counter}")
        folder_location = base_folder
        if folder_location.exists():
            counter += 1
            folder_location = Path(f"metrics/plots{counter}")

    folder_location.mkdir(parents=True, exist_ok=True)
    file_location = Path(folder_location / file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # reading the csv file
    df = pd.read_csv(file_location) 

    # plotting the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')  
    plt.title('Training and Validation Box Loss over Epochs')
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/box_loss.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='blue')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Class Loss')    
    plt.title('Training and Validation Class Loss over Epochs')
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/class_loss.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/dfl_loss'], label='Train dfl Loss', color='blue')
    plt.plot(df['epoch'], df['val/dfl_loss'], label='Val dfl Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('dfl Loss')
    plt.title('Training and Validation dfl Loss over Epochs')
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/dfl_loss.png')
    plt.show()
    plt.close()

    x = np.arange(len(df['epoch']))  
    bar_width = 0.2
    metrics = [
        ('metrics/mAP50(B)', 'mAP50', 'green'),
        ('metrics/mAP50-95(B)', 'mAP50-95', 'orange'),
        ('metrics/precision(B)', 'Precision', 'purple'),
        ('metrics/recall(B)', 'Recall', 'brown')
    ]
    for i, (col, label, color) in enumerate(metrics):
        plt.bar(x+i*bar_width, df[col], width=bar_width, label=label, color=color)
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('mAP, Precision, and Recall over Epochs')
    plt.xticks(x+bar_width*(len(metrics)-1)/2, df['epoch'])
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/mAP_precision_recall_bar.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['epoch'], df['lr/pg0'], label='Learning Rate pg0', color='purple', marker='o')
    plt.scatter(df['epoch'], df['lr/pg1'], label='Learning Rate pg1', color='brown', marker='x')
    plt.scatter(df['epoch'], df['lr/pg2'], label='Learning Rate pg2', color='pink', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.legend()
    plt.savefig(f'metrics/plots{counter}/learning_rate.png') 
    plt.show() 
    plt.close()

@app.post("/fine-tuning") # api endpoint
async def fine_tuning(model_name: str, file: UploadFile = File(...)):
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

