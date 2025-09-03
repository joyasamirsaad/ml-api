# inference.py: functions for making predictions with EfficientDet
#from fastcore.dispatch import typedispatch
from plum import dispatch as typedispatch
from effdet.bench import DetBenchPredict
from ensemble_boxes import weighted_boxes_fusion as run_wbf
from typing import List, Tuple
import torch
import numpy as np
from PIL import Image


class EfficientDetInference:
    def __init__(self, model, device, inference_tfms, img_size: int,
                 prediction_confidence_threshold: float = 0.25,
                 wbf_iou_threshold: float = 0.5):
        """
        Args:
            model: Trained EfficientDet model wrapper
            device: Torch device ("cuda" or "cpu")
            inference_tfms: Albumentations transform for preprocessing images
            img_size: Input size used for training
            prediction_confidence_threshold: Score threshold for keeping predictions
            wbf_iou_threshold: IoU threshold for Weighted Boxes Fusion
        """
        self.model = model
        self.device = device
        self.inference_tfms = inference_tfms
        self.img_size = img_size
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.wbf_iou_threshold = wbf_iou_threshold
        self._predict_model = None  # Lazy init

    @typedispatch
    def predict(self, images: List[Image.Image]) -> Tuple[list, list, list]:
        """
        Run inference on a list of PIL images.
        Returns: (bboxes, labels, scores) as lists per image
        """
        image_sizes = [(img.height, img.width) for img in images]

        images_tensor = torch.stack([
            self.inference_tfms(
                image=np.array(img, dtype=np.float32),
                labels=np.ones(1),
                bboxes=np.array([[0, 0, 1, 1]])
            )["image"]
            for img in images
        ])

        return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    def predict(self, images_tensor: torch.Tensor) -> Tuple[list, list, list]:
        """
        Run inference on batched tensor images (N, 3, H, W).
        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)

        if (images_tensor.shape[-1] != self.img_size or
            images_tensor.shape[-2] != self.img_size):
            raise ValueError(
                f"Expected tensors of shape (N, 3, {self.img_size}, {self.img_size})"
            )

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)


    def _init_predict_model(self):
        if self._predict_model is None:
            self._predict_model = DetBenchPredict(
                self.model.model, self.model.config
            ).to(self.device)
            self._predict_model.eval()

    def _run_inference(self, images_tensor, image_sizes):
        self._init_predict_model()

        detections = self._predict_model(images_tensor.to(self.device))

        bboxes, scores, labels = self._postprocess_detections(detections)
        scaled_bboxes = self._rescale_bboxes(bboxes, image_sizes)

        return scaled_bboxes, labels, scores

    def _postprocess_detections(self, detections):
        preds = []
        for i in range(detections.shape[0]):
            preds.append(self._filter_single_prediction(detections[i]))

        bboxes, scores, labels = run_wbf(
            preds,
            image_size=self.img_size,
            iou_thr=self.wbf_iou_threshold
        )
        return bboxes, scores, labels

    def _filter_single_prediction(self, detections):
        det = detections.detach().cpu().numpy()
        boxes, scores, classes = det[:, :4], det[:, 4], det[:, 5].astype(int)

        keep = np.where(scores > self.prediction_confidence_threshold)[0]
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "classes": classes[keep],
        }

    def _rescale_bboxes(self, bboxes, image_sizes):
        scaled = []
        for boxes, (im_h, im_w) in zip(bboxes, image_sizes):
            if len(boxes) > 0:
                boxes = np.array(boxes) * [
                    im_w / self.img_size,
                    im_h / self.img_size,
                    im_w / self.img_size,
                    im_h / self.img_size,
                ]
                scaled.append(boxes.tolist())
            else:
                scaled.append([])
        return scaled
