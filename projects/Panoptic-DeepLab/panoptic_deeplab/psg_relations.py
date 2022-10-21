import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Transformer
from detectron2.utils.registry import Registry
from detectron2.config import configurable

__all__ = ["PSGRelationNet", "build_psg_relation_net", "PSGRelationEncoder", "build_psg_relation_encoder"]

PSG_RELATION_NET_REGISTRY = Registry("PSG_RELATION_NET")
PSG_RELATION_NET_REGISTRY.__doc__ = """
Registry for psg relations
"""


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_cls = nn.Linear(dim, dim)
        self.proj_rel = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, Q_cls, Q_rel):
        """
        inputs:
        x is the output of transformer encoder
        Q is the fixed query we initialized

        outputs:
        x is the after-attention result
        matt is the attention matrix which will be used as mask prediction
        """
        B, N, C = x.shape
        kv = (
            self.qkv(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        # QK
        matt_cls = (Q_cls @ k.transpose(-2, -1)) * self.scale  # 133*5376
        matt_rel = (Q_rel @ k.transpose(-2, -1)) * self.scale  # 56*5376

        # attention
        attn_cls = matt_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_rel = matt_rel.softmax(dim=-1)
        attn_rel = self.attn_drop(attn_rel)

        # QKV
        x_cls = (attn_cls @ v).transpose(1, 2).reshape(B, -1, C)
        x_cls = self.proj_cls(x_cls)
        x_cls = self.proj_drop(x_cls)

        x_rel = (attn_rel @ v).transpose(1, 2).reshape(B, -1, C)
        x_rel = self.proj_rel(x_rel)
        x_rel = self.proj_drop(x_rel)

        return x_rel, x_cls, matt_cls


def build_psg_relation_net(cfg, input_shape):
    """
    Build a psg relation net from `cfg.MODEL.PSG_RELATION_NET.NAME`.
    """
    name = cfg.MODEL.PSG_RELATION_NET.NAME
    return PSG_RELATION_NET_REGISTRY.get(name)(cfg, input_shape)


@PSG_RELATION_NET_REGISTRY.register()
class PSGRelationNet(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape,
        *,
        feature_name_list,
        encoder_depth,
        total_relations,
        total_classes,
        embed_dim,
        alpha,
        cls_loss_weight=0.5,
        loss_weight=1,
    ):
        super().__init__()
        self.encoder_depth = encoder_depth
        self.total_relations = total_relations
        self.total_classes = total_classes
        self.cls_loss_weight = cls_loss_weight
        self.loss_weight = loss_weight
        self.alpha = alpha

        # ----query
        self.Q_cls = nn.Parameter(
            torch.randn(self.total_classes, embed_dim), requires_grad=False
        )  # fixed query
        self.Q_rel = nn.Parameter(
            torch.randn(self.total_relations, embed_dim), requires_grad=False
        )  # fixed query

        # ----channel mapper
        self.feature_name_list = feature_name_list
        self.map_dict = nn.ModuleDict()
        for k, v in input_shape.items():
            if k in feature_name_list:
                self.map_dict[k] = nn.Linear(v.channels, embed_dim, bias=True)

        # transformer encoder
        self.tf_encoder = Transformer(
            depth=self.encoder_depth,
            num_heads=1,
            embed_dim=embed_dim,
            mlp_ratio=3,
            num_patches=8000,
            drop_rate=0.4,
            attn_drop_rate=0.4
        )

        # transformer decoder
        self.tf_decoder = Attention(
            dim=embed_dim,
            num_heads=1,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.4,
            proj_drop=0.4,
        )

        ## class head
        self.cls_fc = nn.Linear(embed_dim, 1)
        
        # relation head
        self.rel_fc = nn.Linear(embed_dim, 1)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = dict(
            input_shape={
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.PSG_RELATION_NET.IN_FEATURES
            },
            feature_name_list=cfg.MODEL.PSG_RELATION_NET.FEATURE_NAME_LIST,
            encoder_depth=cfg.MODEL.PSG_RELATION_NET.ENCODER_DEPTH,
            total_relations=cfg.MODEL.PSG_RELATION_NET.TOTAL_RELATIONS,
            total_classes=cfg.MODEL.PSG_RELATION_NET.TOTAL_CLASSES,
            embed_dim=cfg.MODEL.PSG_RELATION_NET.EMBED_DIM,
            cls_loss_weight=cfg.MODEL.PSG_RELATION_NET.CLS_LOSS_WEIGHT,
            loss_weight=cfg.MODEL.PSG_RELATION_NET.LOSS_WEIGHT,
            alpha=cfg.MODEL.PSG_RELATION_NET.ALPHA
        )
        return ret

    def forward(self, features, target=None, semantic_cls=None):
        x = self.mapping(features)

        if self.training: 
            if self.alpha > 0:
                out_rel_1, out_cls_1 = self.layers(x)
                out_rel_2, out_cls_2 = self.layers(x)
                out_rel = torch.cat([out_rel_1, out_rel_2], dim=0)
                out_cls = torch.cat([out_cls_1, out_cls_2], dim=0)
                target = torch.cat([target, target.clone()], dim=0)
                semantic_cls = torch.cat([semantic_cls, semantic_cls.clone()], dim=0)
                return None, self.relation_loss_reg(out_rel, out_cls, target, semantic_cls)
            else:
                out_rel, out_cls = self.layers(x)
                return None, self.relation_loss(out_rel, out_cls, target, semantic_cls)
        else:
            out_rel, out_cls = self.layers(x)
            return out_rel, {}

    def mapping(self, features):
        outs = []
        for name, feature in features.items():
            if name in self.feature_name_list:
                b, c, _, _ = feature.shape
                feature = torch.reshape(feature, (b, c, -1))
                feature = feature.permute(0, 2, 1)
                out = self.map_dict[name](feature)
                outs.append(out)

        # concat and feed into transformer encoder
        out = torch.cat(outs, dim=1)
        return out

    def layers(self, x):
        # map to embed dim channel
        x = self.tf_encoder(x)

        # print(out.shape)
        # transformer decoder
        out_rel, out_cls, _ = self.tf_decoder(x, self.Q_cls, self.Q_rel)

        # class head
        out_cls = self.cls_fc(out_cls)

        # relation head
        out_rel = self.rel_fc(out_rel)

        return out_rel.squeeze(-1), out_cls.squeeze(-1)

    def relation_loss(self, out_rel, out_cls, target, semantic_classes):
        loss_rel = F.binary_cross_entropy_with_logits(out_rel, target, reduction="mean")
        loss_cls = F.binary_cross_entropy_with_logits(
            out_cls, semantic_classes, reduction="mean"
        )

        loss = loss_rel + loss_cls * self.cls_loss_weight

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

    def relation_loss_reg(self, out_rel, out_cls, target, semantic_classes):
        loss_rel = F.binary_cross_entropy_with_logits(out_rel, target, reduction="mean")
        loss_cls = F.binary_cross_entropy_with_logits(
            out_cls, semantic_classes, reduction="mean"
        )

        kl_rel_loss = self._compute_kl_loss(out_rel)
        kl_cls_loss = self._compute_kl_loss(out_cls)
        loss_rel += self.alpha * kl_rel_loss
        loss_cls += self.alpha * kl_cls_loss

        loss = loss_rel + loss_cls * self.cls_loss_weight

        losses = {"loss_relation": loss * self.loss_weight}
        return losses


PSG_RELATION_ENCODER_REGISTRY = Registry("PSG_RELATION_ENCODER")
PSG_RELATION_ENCODER_REGISTRY.__doc__ = """
Registry for psg relations encoder
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
        num_heads = 1,
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
            num_heads=num_heads,
            embed_dim=height*width,
            mlp_ratio=3,
            num_patches=self.total_classes,
            drop_rate=0.4,
            attn_drop_rate=0.4
        )

        self.sub_drop = nn.Dropout(0.4)
        self.sub_relation = nn.Linear(height*width, total_relations)
        # self.sub_norm = nn.LayerNorm(56)
        self.obj_drop = nn.Dropout(0.4)
        self.obj_relation = nn.Linear(height*width, total_relations)
        # self.obj_norm = nn.LayerNorm(56)

    @classmethod
    def from_config(cls, cfg):
        ret = dict(
            encoder_depth=cfg.MODEL.PSG_RELATION_ENCODER.ENCODER_DEPTH,
            total_relations=cfg.MODEL.PSG_RELATION_ENCODER.TOTAL_RELATIONS,
            total_classes=cfg.MODEL.PSG_RELATION_ENCODER.TOTAL_CLASSES,
            height=cfg.MODEL.PSG_RELATION_ENCODER.HEIGHT,
            width=cfg.MODEL.PSG_RELATION_ENCODER.WIDTH,
            num_heads=cfg.MODEL.PSG_RELATION_ENCODER.NUM_HEADS,
            alpha=cfg.MODEL.PSG_RELATION_ENCODER.ALPHA,
            loss_weight=cfg.MODEL.PSG_RELATION_ENCODER.LOSS_WEIGHT,
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
            rel = torch.sum(rel, dim=1)

            return rel, {}


    def layers(self, x):
        # map to embed dim channel
        # x [B, C, H, W]
        x = self.avgPool(x) # x [B, C, h, w]
        x = torch.flatten(x, 2, 3) # x [B, C, T] C == num of patches
        x = self.tf_encoder(x)

        # class head
        sub_rel =  self.sub_relation(self.sub_drop(x))

        # relation head
        obj_rel = self.obj_relation(self.obj_drop(x))

        return sub_rel, obj_rel

    def relation_loss(self, sub_rel, obj_rel, sub_target, obj_target):
        loss_sub = F.binary_cross_entropy_with_logits(sub_rel, sub_target, reduction="mean")
        loss_obj = F.binary_cross_entropy_with_logits(
            obj_rel, obj_target, reduction="mean"
        )

        loss = loss_sub + loss_obj

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
        p_loss = p_loss.mean() * 0.5
        q_loss = q_loss.mean() * 0.5

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