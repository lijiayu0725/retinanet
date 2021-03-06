import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_

from anchor import AnchorGenerator
from anchor import anchor_target
from loss import FocalLoss, SmoothL1Loss


class RetinaHead(nn.Module):

    def __init__(self,
                 num_classes=81,
                 in_channels=256,
                 stacked_convs=4,
                 anchor_ratios=(0.5, 1, 2),
                 anchor_strides=(8, 16, 32, 64, 128)):
        super(RetinaHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.anchor_scales = np.array([2 ** (i / 3) for i in range(3)]) * 4
        self.cls_out_channels = num_classes - 1
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(anchor_strides)
        self.loss_cls = FocalLoss()
        self.loss_bbox = SmoothL1Loss()
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(AnchorGenerator(anchor_base, self.anchor_scales, anchor_ratios))
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            self.cls_convs.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))
            self.reg_convs.append(nn.Conv2d(256, 256, 3, stride=1, padding=1))

        self.retina_cls = nn.Conv2d(256, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.retina_reg = nn.Conv2d(256, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_(m.weight, std=0.01)
        for m in self.reg_convs:
            normal_(m.weight, std=0.01)
        normal_(self.retina_cls.weight, std=0.01)
        normal_(self.retina_reg.weight, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            unmap_outputs=True
        )
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = num_total_pos
        losses_cls, losses_bbox = [], []
        for i in range(len(featmap_sizes)):
            loss_cls, loss_bbox = self.loss_single(
                cls_scores[i],
                bbox_preds[i],
                labels_list[i],
                label_weights_list[i],
                bbox_targets_list[i],
                bbox_weights_list[i],
                num_total_samples=num_total_samples
            )
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)
        return dict(loss_cls=tuple(losses_cls), loss_bbox=tuple(losses_bbox))

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for i in range(len(feats)):
            score, pred = self.forward_single(feats[i])
            cls_scores.append(score)
            bbox_preds.append(pred)
        return cls_scores, bbox_preds
