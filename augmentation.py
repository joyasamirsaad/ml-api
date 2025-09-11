from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
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
