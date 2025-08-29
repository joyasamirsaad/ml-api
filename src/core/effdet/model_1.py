# 6. model_1.py: defines a PyTorch Lightning Module that encapsulates the EfficientDet model, training, and validation logic.
import torch
from pytorch_lightning import LightningModule
from core.effdet.model import create_model
from core.effdet.transformations import get_valid_transforms


class EfficientDetModel(LightningModule):
    def __init__(
        self,
        num_classes=1,
        img_size=512,
        prediction_confidence_threshold=0.2,
        learning_rate=0.0002,
        wbf_iou_threshold=0.44,
        inference_transforms=get_valid_transforms(target_img_size=512),
        model_architecture='tf_efficientnetv2_l',
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    # training_step and validation_step remain unchanged



    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        losses = self.model(images, targets)

        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_class_loss", losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return losses["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        outputs = self.model(images, targets)

        batch_predictions = {
            "predictions": outputs["detections"],
            "targets": targets,
            "image_ids": image_ids,
        }

        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("valid_class_loss", outputs["class_loss"].detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("valid_box_loss", outputs["box_loss"].detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {"loss": outputs["loss"], "batch_predictions": batch_predictions}
