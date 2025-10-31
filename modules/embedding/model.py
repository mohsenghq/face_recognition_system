# model.py
import torch
import torch.nn as nn
from modules.embedding.net import build_model as build_backbone
from modules.embedding.head import build_head


class FullModel(nn.Module):
    """Wraps backbone + head for training/inference."""
    def __init__(self, model_name='ir_101', head_type='adaface',
                 embedding_size=512, class_num=70722,
                 m=0.4, h=0.333, s=64., t_alpha=1.0):
        super().__init__()
        self.backbone = build_backbone(model_name)
        self.head = build_head(head_type,
                               embedding_size=embedding_size,
                               class_num=class_num,
                               m=m, h=h, s=s, t_alpha=t_alpha)

    def forward(self, x, label=None):
        # backbone returns both embedding and norm
        embedding, norm = self.backbone(x)
        # during inference you usually don't pass labels
        if label is None:
            return embedding
        logits = self.head(embedding, norm, label)
        return logits, embedding

class InferenceModel(nn.Module):
    def __init__(self, model_name='ir_101'):
        super().__init__()
        self.backbone = build_backbone(model_name)

    def forward(self, x):
        embedding, _ = self.backbone(x)
        return embedding

def build_model(model_name='ir_101'):
    """
    Returns only the backbone for inference (no head),
    matching your current pattern.
    """
    return InferenceModel(model_name)
