# 7. model_predict.py: prediction
#from fastcore.dispatch import typedispatch
from plum import dispatch as typedispatch
from typing import List, Tuple
import torch
import numpy as np
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion as run_wbf
from effdet.bench import DetBenchPredict

class EfficientDetModelMixin:
    @typedispatch
    def predict(self, images: List[Image.Image]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Make predictions on a list of PIL images.
        
        Returns:
            tuple of lists: (bboxes_list, class_labels_list, confidences_list)
        """
        self._predict_model=None
        self.eval()
        device = next(self.model.parameters()).device

        # original image sizes
        image_sizes = [(img.height, img.width) for img in images]

        # Apply inference transforms and stack into tensor
        images_tensor = torch.stack([
            self.inference_tfms(
                image=np.array(img, dtype=np.float32),
                bboxes=np.array([[0, 0, 1, 1]]),  
                labels=np.ones(1)                 
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
    
    def _init_predict_model(self):
        if self._predict_model is None:
            self._predict_model = DetBenchPredict(
                self.model.model
            ).to(self.device)
            self._predict_model.eval()

    def _run_inference(self, images_tensor, image_sizes):
        self._init_predict_model()

        detections = self._predict_model(images_tensor.to(self.device))

        bboxes, scores, labels = self._postprocess_detections(detections)
        scaled_bboxes = self._rescale_bboxes(bboxes, image_sizes)

        return scaled_bboxes, labels, scores

    def _postprocess_detections(self, detections):
        all_bboxes, all_scores, all_labels = [],[],[]
        for i in range(detections.shape[0]):
            pred = self._filter_single_prediction(detections[i])

            # Convert boxes to normalized coordinates (0-1) for WBF
            normalized_boxes = pred["boxes"] / self.img_size
            
            bboxes, scores, labels = run_wbf(
                [normalized_boxes.tolist()],
                [pred["scores"].tolist()],
                [pred["classes"].tolist()],
                weights=[1],  # single model
                iou_thr=self.wbf_iou_threshold,
                skip_box_thr=self.prediction_confidence_threshold
            )
            
            # Convert back to pixel coordinates
            bboxes = np.array(bboxes) * self.img_size if len(bboxes) > 0 else np.array([])
            
            all_bboxes.append(bboxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_bboxes, all_scores, all_labels

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