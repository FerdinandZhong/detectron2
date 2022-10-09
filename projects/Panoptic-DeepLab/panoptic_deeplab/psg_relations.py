import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Transformer
from detectron2.utils.registry import Registry
from detectron2.config import configurable

__all__ = ["PSGRelationNet", "build_psg_relation_net"]

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
        encoder_depth,
        total_relations,
        total_classes,
        cls_loss_weight=0.5,
        loss_weight=1
    ):
        super().__init__()
        self.encoder_depth = encoder_depth
        self.total_relations = total_relations
        self.total_classes = total_classes
        self.cls_loss_weight = cls_loss_weight
        self.loss_weight = loss_weight

        # ----query
        self.Q_cls = nn.Parameter(
            torch.randn(self.total_classes, 128), requires_grad=False
        )  # fixed query
        self.Q_rel = nn.Parameter(
            torch.randn(self.total_relations, 128), requires_grad=False
        )  # fixed query

        # ----channel mapper

        self.map_dict = nn.ModuleDict()
        for k, v in input_shape.items():
            self.map_dict[k] = nn.Linear(v.channels, 128, bias=True)

        # transformer encoder
        self.tf_encoder = Transformer(
            depth=self.encoder_depth,
            num_heads=1,
            embed_dim=128,
            mlp_ratio=3,
            num_patches=33600,
        )

        # transformer decoder
        self.tf_decoder = Attention(
            dim=128,
            num_heads=1,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
        )

        ## class head
        self.cls_fc = nn.Linear(128, 1)

        # relation head
        self.rel_fc = nn.Linear(128, 1)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = dict(
            input_shape={
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.PSG_RELATION_NET.IN_FEATURES
            },
            encoder_depth=cfg.MODEL.PSG_RELATION_NET.ENCODER_DEPTH,
            total_relations=cfg.MODEL.PSG_RELATION_NET.TOTAL_RELATIONS,
            total_classes=cfg.MODEL.PSG_RELATION_NET.TOTAL_CLASSES,
            cls_loss_weight=cfg.MODEL.PSG_RELATION_NET.CLS_LOSS_WEIGHT,
            loss_weight=cfg.MODEL.PSG_RELATION_NET.LOSS_WEIGHT
        )
        return ret

    def forward(self, features, target=None, semantic_cls=None):
        out_rel, out_cls = self.layers(features)
        if self.training:
            return None, self.relation_loss(out_rel, out_cls, target, semantic_cls)
        else:
            return out_rel, {}

    def layers(self, features):
        # map to 128 channel
        outs = []
        for name, feature in features.items():
            feature = torch.flatten(feature, 2, 3)
            feature = feature.permute(0, 2, 1)
            out = self.map_dict[name](feature)
            outs.append(out)

        # concat and feed into transformer encoder
        out = torch.cat(outs, dim=1)
        out = self.tf_encoder(out)

        # print(out.shape)
        # transformer decoder
        out_rel, out_cls, _ = self.tf_decoder(out, self.Q_cls, self.Q_rel)

        # class head
        out_cls = self.cls_fc(out_cls)

        # relation head
        out_rel = self.rel_fc(out_rel)

        return out_rel.squeeze(-1), out_cls.squeeze(-1)

    def relation_loss(self, out_rel, out_cls, target, semantic_classes):
        loss_rel = F.binary_cross_entropy_with_logits(out_rel, target, reduction="sum")
        loss_cls = F.binary_cross_entropy_with_logits(
            out_cls, semantic_classes, reduction="sum"
        )

        loss = loss_rel + loss_cls * self.cls_loss_weight

        losses = {"loss_relation": loss * self.loss_weight}
        return losses
