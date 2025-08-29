from fastapi import UploadFile
from pathlib import Path
import json
import cv2
import numpy as np


async def heatmap(file: UploadFile):
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
