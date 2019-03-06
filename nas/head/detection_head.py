import torch
import torch.nn as nn

from mmdet.models import build_head, build_roi_extractor, build_neck
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler

from .base_head import BaseHead

class DetectionHead(BaseHead):
  """Head for classification.

  Two stage.
  """

  def __init__(self, cfg, train_cfg,
               test_cfg):
    """Get a head, e.g. RPN
    See mmdetection for details.

    Parameters
    ----------
    cfg : dict
      e.g.     return build(cfg, BACKBONES)
       dict(dict(type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_scales=[8],
            anchor_ratios=[0.5, 1.0, 2.0],
            anchor_strides=[4, 8, 16, 32, 64],
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]))
      see mmdetection for details.

    """

    super(DetectionHead, self).__init__()

    if 'neck' in cfg:
      self.neck = build_neck(cfg['neck'])

    if 'rpn_head' in cfg:
      self.rpn_head = build_head(cfg['rpn_head'])
    
    if 'bbox_head' in cfg:
      self.bbox_roi_extractor = build_roi_extractor(
                            cfg['bbox_roi_extractor'])
      self.bbox_head = build_head(cfg['bbox_head'])
    
    if 'mask_head' in cfg:
      self.mask_roi_extractor = build_roi_extractor(
                            cfg['mask_roi_extractor'])
      self.mask_head = build_head(cfg['mask_head'])
    
    self.train_cfg = train_cfg
    self.test_cfg = test_cfg
  
  @property
  def with_rpn(self):
    return hasattr(self, 'rpn_head') and self.rpn_head is not None
  
  @property
  def with_bbox(self):
    return hasattr(self, 'bbox_head') and self.bbox_head is not None
  
  @property
  def with_mask(self):
    return hasattr(self, 'mask_head') and self.mask_head is not None
  
  @property
  def with_neck(self):
    return hasattr(self, 'neck') and self.neck is not None
  

  def forward(self, 
              x,
              img_meta,
              gt_bboxes,
              gt_bboxes_ignore,
              gt_labels,
              gt_masks=None,
              proposals=None):
    """
    Copy from mmdetection.

    Parameters
    ----------

    """
    losses = dict()
    num_imgs = x[0].size(0)
  
    if self.with_neck:
      x = self.neck(x)

    # RPN forward and loss
    if self.with_rpn:
      rpn_outs = self.rpn_head(x)

      rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                    self.train_cfg['rpn'])
      rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
      losses.update(rpn_losses)

      proposal_inputs = rpn_outs + (img_meta, self.test_cfg['rpn'])
      proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
    else:
      proposal_list = proposals

    # assign gts and sample proposals
    if self.with_bbox or self.with_mask:
      bbox_assigner = build_assigner(self.train_cfg['rcnn']['assigner'])
      bbox_sampler = build_sampler(
          self.train_cfg['rcnn']['sampler'], context=self)
      
      sampling_results = []
      for i in range(num_imgs):
        assign_result = bbox_assigner.assign(
            proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
            gt_labels[i])
        sampling_result = bbox_sampler.sample(
            assign_result,
            proposal_list[i],
            gt_bboxes[i],
            gt_labels[i],
            feats=[lvl_feat[i][None] for lvl_feat in x])
        sampling_results.append(sampling_result)

    # bbox head forward and loss
    if self.with_bbox:
      rois = bbox2roi([res.bboxes for res in sampling_results])
      # TODO: a more flexible way to decide which feature maps to use
      bbox_feats = self.bbox_roi_extractor(
          x[:self.bbox_roi_extractor.num_inputs], rois)
      cls_score, bbox_pred = self.bbox_head(bbox_feats)

      bbox_targets = self.bbox_head.get_target(
          sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)

      loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                      *bbox_targets)
      losses.update(loss_bbox)

    # mask head forward and loss
    if self.with_mask:
      pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
      mask_feats = self.mask_roi_extractor(
          x[:self.mask_roi_extractor.num_inputs], pos_rois)
      mask_pred = self.mask_head(mask_feats)

      mask_targets = self.mask_head.get_target(
          sampling_results, gt_masks, self.train_cfg.rcnn)
      pos_labels = torch.cat(
          [res.pos_gt_labels for res in sampling_results])
      loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                      pos_labels)
      losses.update(loss_mask)

    return losses
