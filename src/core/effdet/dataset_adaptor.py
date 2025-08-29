# 1. dataset_adaptor.py: converts the specific raw dataset format into an image and corresponding annotations. 
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
import pandas as pd


class CarsDatasetAdaptor:
    def __init__(self, images_dir_path: str, annotation_json_path: str):
        self.images_dir_path = Path(images_dir_path)

        # Load COCO-style JSON
        with open(annotation_json_path, "r") as f:
            coco = json.load(f)

        # Convert JSON parts to DataFrames
        images_df = pd.DataFrame(coco["images"])        # id, file_name, width, height
        annotations_df = pd.DataFrame(coco["annotations"])  # image_id, category_id, bbox
        categories_df = pd.DataFrame(coco["categories"])    # id, name

        # Merge annotations with image file names
        self.annotations_df = annotations_df.merge(
            images_df[["id", "file_name"]], left_on="image_id", right_on="id"
        )

        # Keep only useful columns
        self.annotations_df = self.annotations_df[["file_name", "bbox", "category_id"]]

        # List of unique images
        self.images = self.annotations_df.file_name.unique().tolist()

        # Map category_id → category_name
        self.cat_mapping = dict(zip(categories_df.id, categories_df.name))

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = Image.open(self.images_dir_path / image_name).convert("RGB")

        # Extract all bboxes for this image
        bboxes = self.annotations_df[self.annotations_df.file_name == image_name]["bbox"].values
        pascal_bboxes = np.array([[x, y, x + w, y + h] for [x, y, w, h] in bboxes])

        # Extract class labels
        class_labels = self.annotations_df[self.annotations_df.file_name == image_name]["category_id"].values

        return image, pascal_bboxes, class_labels, index

    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")

        # Convert PIL → OpenCV
        image_cv = np.array(image)
        if image_cv.shape[-1] == 3:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        for bbox, label in zip(bboxes, class_labels):
            xmin, ymin, xmax, ymax = map(int, bbox)
            class_name = self.cat_mapping.get(label, str(label))
            cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(image_cv, class_name, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(f"Image ID: {image_id}", image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Classes:", [self.cat_mapping.get(l, l) for l in class_labels])
