from pathlib import Path
from fastapi import UploadFile
import zipfile
import albumentations as A
import cv2

def albumentations_transform(image_path):
    transform = A.Compose([
        #A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.8),
        A.Rotate(limit=45, p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
    ])

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    augmented = transform(image=image)
    augmented_image = augmented['image']
    
    #print("i am here")
    augmentations = image_path.parent
    file_path = augmentations / f"{image_path.stem}_augmented.jpg"
    cv2.imwrite(str(file_path), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

    #print("Augmentation complete.")

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