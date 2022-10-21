import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Transformer
from detectron2.utils.registry import Registry
from detectron2.config import configurable

__all__ = ["PSGRelationEncoder", "build_psg_relation_encoder"]

PSG_RELATION_ENCODER_REGISTRY = Registry("PSG_RELATION_ENCODER")
PSG_RELATION_ENCODER_REGISTRY.__doc__ = """
Registry for psg relations
"""


def build_psg_relation_encoder(cfg):
    """
    Build a psg relation net from `cfg.MODEL.PSG_RELATION_ENCODER.NAME`.
    """
    name = cfg.MODEL.PSG_RELATION_ENCODER.NAME
    return PSG_RELATION_ENCODER_REGISTRY.get(name)(cfg)


@PSG_RELATION_ENCODER_REGISTRY.register()
class PSGRelationEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        encoder_depth,
        total_relations,
        total_classes,
        height,
        width,
        alpha=0,
        loss_weight=1,
    ):
        super().__init__()
        self.encoder_depth = encoder_depth
        self.total_relations = total_relations
        self.total_classes = total_classes
        self.loss_weight = loss_weight
        self.alpha = alpha

        self.avgPool = nn.AdaptiveAvgPool2d((height, width))
        # transformer encoder
        self.tf_encoder = Transformer(
            depth=self.encoder_depth,
            num_heads=1,
            embed_dim=height*width,
            mlp_ratio=3,
            num_patches=self.total_classes,
            drop_rate=0.4,
            attn_drop_rate=0.4
        )

        self.sub_relation = nn.Linear(height*width, total_relations)
        self.obj_relation = nn.Linear(height*width, total_relations)

    @classmethod
    def from_config(cls, cfg):
        ret = dict(
            encoder_depth=cfg.MODEL.PSG_RELATION_ENCODER.ENCODER_DEPTH,
            total_relations=cfg.MODEL.PSG_RELATION_ENCODER.TOTAL_RELATIONS,
            total_classes=cfg.MODEL.PSG_RELATION_ENCODER.TOTAL_CLASSES,
            height=cfg.MODEL.PSG_RELATION_ENCODER.HEIGHT,
            width=cfg.MODEL.PSG_RELATION_ENCODER.WIDTH,
            loss_weight=cfg.MODEL.PSG_RELATION_ENCODER.LOSS_WEIGHT,
            alpha=cfg.MODEL.PSG_RELATION_ENCODER.ALPHA
        )
        return ret

    def forward(self, x, sub_target=None, obj_target=None):
        if self.training: 
            if self.alpha > 0:
                sub_rel_1, obj_rel_1 = self.layers(x)
                sub_rel_2, obj_rel_2 = self.layers(x)
                sub_rel = torch.cat([sub_rel_1, sub_rel_2], dim=0)
                obj_rel = torch.cat([obj_rel_1, obj_rel_2], dim=0)
                sub_target = torch.cat([sub_target, sub_target.clone()], dim=0)
                obj_target = torch.cat([obj_target, obj_target.clone()], dim=0)
                return None, self.relation_loss_reg(sub_rel, obj_rel, sub_target, obj_target)
            else:
                sub_rel, obj_rel = self.layers(x)
                return None, self.relation_loss(sub_rel, obj_rel, sub_target, obj_target)
        else:
            sub_rel, obj_rel = self.layers(x)
            rel = torch.cat([sub_rel, obj_rel], dim=1)
            rel = torch.mean(rel, dim=1)

            return rel, {}


    def layers(self, x):
        # map to embed dim channel
        # x [B, C, H, W]
        x = self.avgPool(x) # x [B, C, h, w]

        x = torch.flatten(x, 2, 3) # x [B, C, T] C == num of patches

        x = self.tf_encoder(x)

        print(x.shape)

        # class head
        sub_rel = self.sub_relation(x)

        # relation head
        obj_rel = self.obj_relation(x)

        return sub_rel.squeeze, obj_rel.squeeze

    def relation_loss(self, sub_rel, obj_rel, sub_target, obj_target):
        loss_sub = F.binary_cross_entropy_with_logits(sub_rel, sub_target, reduction="mean")
        loss_obj = F.binary_cross_entropy_with_logits(
            obj_rel, obj_target, reduction="mean"
        )

        loss = loss_sub + loss_obj * self.cls_loss_weight

        losses = {"loss_relation": loss * self.loss_weight}
        return losses
    
    def _compute_kl_loss(self, logits):
        p,  q = torch.split(logits, logits.size(0)//2, dim=0)
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none"
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none"
        )

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = p_loss + q_loss

        return loss

    def relation_loss_reg(self, sub_rel, obj_rel, sub_target, obj_target):
        loss_sub = F.binary_cross_entropy_with_logits(sub_rel, sub_target, reduction="mean")
        loss_obj = F.binary_cross_entropy_with_logits(
            obj_rel, obj_target, reduction="mean"
        )

        kl_sub_loss = self._compute_kl_loss(sub_rel)
        kl_obj_loss = self._compute_kl_loss(obj_rel)
        loss_sub += self.alpha * kl_sub_loss
        loss_obj += self.alpha * kl_obj_loss

        loss = loss_sub + loss_obj

        losses = {"loss_relation": loss * self.loss_weight}
        return losses

