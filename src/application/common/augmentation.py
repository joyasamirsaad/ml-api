from pathlib import Path
from fastapi import UploadFile
import zipfile
from augmentation import albumentations_transform


async def augmentation(file: UploadFile):
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