import torch
from torch import nn

from fpn import FPN
from head import RetinaHead
from resnet import ResNet


class RetinaNet(nn.Module):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self, pretrained=None):
        super(RetinaNet, self).__init__()
        self.backbone = ResNet()
        self.neck = FPN()
        self.bbox_head = RetinaHead()
        self.init_weights()

    def init_weights(self):
        self.backbone.load_state_dict(
            torch.load('/Users/nick/.cache/torch/checkpoints/resnet50-19c8e357.pth', strict=False))
        self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward(self, img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        return self.forward_train(img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None)
