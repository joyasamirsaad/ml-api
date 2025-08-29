# 7. model_predict.py: prediction
#from fastcore.dispatch import typedispatch
from plum import dispatch as typedispatch
from typing import List, Tuple
import torch
import numpy as np
from PIL import Image

class EfficientDetModelMixin:
    @typedispatch
    def predict(self, images: List[Image.Image]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Make predictions on a list of PIL images.
        
        Returns:
            tuple of lists: (bboxes_list, class_labels_list, confidences_list)
        """
        self.eval()
        device = next(self.model.parameters()).device

        # Record original image sizes
        image_sizes = [(img.height, img.width) for img in images]

        # Apply inference transforms and stack into tensor
        images_tensor = torch.stack([
            self.inference_tfms(
                image=np.array(img, dtype=np.float32),
                bboxes=np.array([[0, 0, 1, 1]]),  # dummy bbox
                labels=np.ones(1)                 # dummy label
            )['image']
            for img in images
        ]).to(device)

        return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    def predict(self, images_tensor: torch.Tensor) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Make predictions on a tensor from the dataloader.
        
        Args:
            images_tensor: shape (N, 3, H, W) or (3, H, W)
        
        Returns:
            tuple of lists: (bboxes_list, class_labels_list, confidences_list)
        """
        self.eval()
        device = next(self.model.parameters()).device

        # If single image, add batch dimension
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)

        # Check size
        if images_tensor.shape[-2] != self.img_size or images_tensor.shape[-1] != self.img_size:
            raise ValueError(f"Expected images of shape (N,3,{self.img_size},{self.img_size})")

        images_tensor = images_tensor.to(device)
        image_sizes = [(self.img_size, self.img_size)] * images_tensor.shape[0]

        return self._run_inference(images_tensor, image_sizes)
