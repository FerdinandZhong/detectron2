# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config


def add_panoptic_deeplab_config(cfg):
    """
    Add config for Panoptic-DeepLab.
    """
    # Reuse DeepLab config.
    add_deeplab_config(cfg)
    # Target generation parameters.
    cfg.INPUT.GAUSSIAN_SIGMA = 10
    cfg.INPUT.IGNORE_STUFF_IN_OFFSET = True
    cfg.INPUT.SMALL_INSTANCE_AREA = 4096
    cfg.INPUT.SMALL_INSTANCE_WEIGHT = 3
    cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC = False
    # Optimizer type.
    cfg.SOLVER.OPTIMIZER = "ADAM"
    # Panoptic-DeepLab semantic segmentation head.
    # We add an extra convolution before predictor.
    cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K = 0.2
    # Panoptic-DeepLab instance segmentation head.
    cfg.MODEL.INS_EMBED_HEAD = CN()
    cfg.MODEL.INS_EMBED_HEAD.NAME = "PanopticDeepLabInsEmbedHead"
    cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
    cfg.MODEL.INS_EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
    cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
    cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT = 0.1
    # We add an extra convolution before predictor.
    cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS = 32
    cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM = 128
    cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.INS_EMBED_HEAD.NORM = "SyncBN"
    cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT = 200.0
    cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT = 0.01
    # Panoptic-DeepLab post-processing setting.
    cfg.MODEL.PANOPTIC_DEEPLAB = CN()
    # Stuff area limit, ignore stuff region below this number.
    cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA = 2048
    cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD = 0.1
    cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL = 7
    cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE = 200
    # If set to False, Panoptic-DeepLab will not evaluate instance segmentation.
    cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES = True
    cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV = False
    # This is the padding parameter for images with various sizes. ASPP layers
    # requires input images to be divisible by the average pooling size and we
    # can use `MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY` to pad all images to
    # a fixed resolution (e.g. 640x640 for COCO) to avoid having a image size
    # that is not divisible by ASPP average pooling size.
    cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY = -1
    # Only evaluates network speed (ignores post-processing).
    cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED = False

    # for relations psg
    # cfg.MODEL.PSG_RELATION_NET = CN()
    # cfg.MODEL.PSG_RELATION_NET.NAME = "PSGRelationNet"
    # cfg.MODEL.PSG_RELATION_NET.IN_FEATURES = ["res3", "res5"]
    # cfg.MODEL.PSG_RELATION_NET.ENCODER_DEPTH = 2
    # cfg.MODEL.PSG_RELATION_NET.TOTAL_RELATIONS = 56
    # cfg.MODEL.PSG_RELATION_NET.TOTAL_CLASSES = 133
    # cfg.MODEL.PSG_RELATION_NET.EMBED_DIM = 256
    # cfg.MODEL.PSG_RELATION_NET.FEATURE_NAME_LIST = ["res3", "res5"]
    # cfg.MODEL.PSG_RELATION_NET.LOSS_WEIGHT = 1.0
    # cfg.MODEL.PSG_RELATION_NET.CLS_LOSS_WEIGHT = 1.0
    # cfg.MODEL.PSG_RELATION_NET.ALPHA = 0.0

    # for relations encoder
    cfg.MODEL.PSG_RELATION_ENCODER = CN()
    cfg.MODEL.PSG_RELATION_ENCODER.NAME = "PSGRelationEncoder"
    cfg.MODEL.PSG_RELATION_ENCODER.ENCODER_DEPTH = 2
    cfg.MODEL.PSG_RELATION_ENCODER.TOTAL_RELATIONS = 56
    cfg.MODEL.PSG_RELATION_ENCODER.TOTAL_CLASSES = 133
    cfg.MODEL.PSG_RELATION_ENCODER.HEIGHT = 80
    cfg.MODEL.PSG_RELATION_ENCODER.WIDTH = 80
    cfg.MODEL.PSG_RELATION_ENCODER.NUM_HEADS = 1
    cfg.MODEL.PSG_RELATION_ENCODER.ALPHA = 0.0
    cfg.MODEL.PSG_RELATION_ENCODER.LOSS_WEIGHT = 1.0
