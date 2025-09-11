from pathlib import Path
from ultralytics import YOLO
 
def inference(image_path: Path):
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
        "image_path": str(f"detection_{image_path}"),
        "objects_detected": objects
    }

    return { "detection_result": detection_result }