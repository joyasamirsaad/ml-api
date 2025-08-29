# 6. inference.py: contains functions for making predictions using the trained model, handling both image inputs and tensor inputs.
from fastcore.dispatch import typedispatch
from typing import List
import torch
import numpy as np

@typedispatch
def predict(self, images: List):
    """
    For making predictions from images
    Args:
        images: a list of PIL images

    Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

    """
    image_sizes = [(image.size[1], image.size[0]) for image in images]
    images_tensor = torch.stack(
        [
            self.inference_tfms(
                image=np.array(image, dtype=np.float32),
                labels=np.ones(1),
                bboxes=np.array([[0, 0, 1, 1]]),
            )["image"]
            for image in images
        ]
    )

    return self._run_inference(images_tensor, image_sizes)

@typedispatch
def predict(self, images_tensor: torch.Tensor):
    """
    For making predictions from tensors returned from the model's dataloader
    Args:
        images_tensor: the images tensor returned from the dataloader

    Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

    """
    if images_tensor.ndim == 3:
        images_tensor = images_tensor.unsqueeze(0)
    if (
        images_tensor.shape[-1] != self.img_size
        or images_tensor.shape[-2] != self.img_size
    ):
        raise ValueError(
            f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})"
        )

    num_images = images_tensor.shape[0]
    image_sizes = [(self.img_size, self.img_size)] * num_images

    return self._run_inference(images_tensor, image_sizes)