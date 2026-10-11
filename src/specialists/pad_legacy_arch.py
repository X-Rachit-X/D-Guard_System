"""
src/models/architectures.py - Network definitions for the trained DERMA-GUARD backends.

These are vendored (not imported from the Dummy/ tree) so the root project is a
self-contained, inspectable codebase; only the *weights* are loaded from the
artifact directory. Every layer name and shape here is load-bearing: the
checkpoints are keyed by attribute path, so renaming a submodule silently
degrades a model to random initialisation.

src/models/backends.py loads each of these with strict=True precisely so that
any drift between this file and the checkpoints fails loudly instead of quietly.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class DermatologicalChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention with signed-sqrt power normalisation."""

    def __init__(self, in_features: int = 2048, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features // reduction)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features // reduction, in_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pow = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-7)
        x_norm = F.normalize(x_pow, p=2, dim=-1)
        att = self.sigmoid(self.fc2(self.act(self.fc1(x_norm))))
        return x_norm * (1.0 + att)


class MHSATransformerBlock(nn.Module):
    """Pre-LayerNorm multi-head self-attention block with a residual FFN."""

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                         dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + self.dropout1(attn_out)
        return x + self.ffn(self.norm2(x))


# ---------------------------------------------------------------------------
# Modality encoders
# ---------------------------------------------------------------------------

class VisionTransformerImageEncoder(nn.Module):
    """ResNet50 features -> channel attention -> visual tokens -> MHSA -> 256-d.

    Accepts either a raw image batch (B,3,224,224) or a pre-extracted 2048-d
    ResNet50 feature batch, which is how the cached-feature training path used it.
    """

    def __init__(self, embed_dim: int = 256, num_tokens: int = 8, num_heads: int = 8,
                 num_layers: int = 2, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = tvm.resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_features = 2048
        self.attention = DermatologicalChannelAttention(in_features=in_features, reduction=16)

        self.token_proj = nn.Sequential(
            nn.Linear(in_features, num_tokens * embed_dim),
            nn.LayerNorm(num_tokens * embed_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + 1, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            MHSATransformerBlock(d_model=embed_dim, nhead=num_heads,
                                 dim_feedforward=embed_dim * 2, dropout=0.10)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        # Kept because the checkpoints contain it; used for 2048-d shortcut paths.
        self.projection = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(384, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x) if x.dim() == 4 else x
        b = feats.size(0)
        feats = self.attention(feats)
        tokens = self.token_proj(feats).view(b, self.num_tokens, self.embed_dim)
        tokens = torch.cat([self.cls_token.expand(b, -1, -1), tokens], dim=1) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(self.norm(tokens[:, 0, :]))


ImageEncoder = VisionTransformerImageEncoder


class MetaBlockMLPEncoder(nn.Module):
    """Deep residual MLP over engineered tabular metadata (116-d -> 256-d)."""

    def __init__(self, input_dim: int = 116, embed_dim: int = 256,
                 hidden_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.gelu = nn.GELU()
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.input_layer(x)
        h1 = self.gelu(h0 + self.res_block(h0))
        return self.output_layer(h1)


MetadataEncoder = MetaBlockMLPEncoder


class LanguageEncoder(nn.Module):
    """Bio_ClinicalBERT -> 768-d [CLS] -> 256-d projection.

    NOTE: the trained projection expects the [CLS] token, not a mean-pool. The
    DERMA-GUARD spec (proj.md Part 3) calls for mean-pooling; feeding mean-pooled
    vectors here would put the projection off-distribution. The deviation is
    deliberate and recorded in docs/evaluation_integrity.md.
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 embed_dim: int = 256, load_bert: str | bool = "config",
                 freeze_backbone: bool = True):
        """
        load_bert:
            "config"     - build the BERT graph from its config with random weights.
                           The specialist checkpoints contain the (frozen) BERT
                           tensors, so a strict load restores them exactly. This is
                           the default: it keeps loading offline and lets
                           strict=True actually verify the whole graph.
            "pretrained" - download/read the pretrained weights (for encoding raw
                           text outside a specialist checkpoint).
            False        - no BERT at all; only the 768-d projection is built.
        """
        super().__init__()
        self.bert = None
        if load_bert == "pretrained":
            from transformers import AutoModel
            try:
                self.bert = AutoModel.from_pretrained(model_name)
            except Exception:
                self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        elif load_bert == "config":
            from transformers import AutoConfig, AutoModel
            try:
                cfg = AutoConfig.from_pretrained(model_name)
            except Exception:
                cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
            self.bert = AutoModel.from_config(cfg)
        if self.bert is not None and freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(384, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
        )

    def forward(self, input_ids=None, attention_mask=None, text_embed=None):
        if text_embed is not None:
            return self.projection(text_embed) if text_embed.shape[-1] == 768 else text_embed
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.projection(outputs.last_hidden_state[:, 0, :])


class TextResidualProjector(nn.Module):
    """Standalone 768-d -> 256-d residual projector (saved as text_proj.pt).

    This is the architecture the training run actually saved. Dummy's own
    predict.py rebuilt it as a plain nn.Sequential and loaded with strict=False,
    which silently discarded every weight; see docs/evaluation_integrity.md.
    """

    def __init__(self, in_d: int = 768, out_d: int = 256):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(0.1),
        )
        self.res = nn.Sequential(
            nn.Linear(out_d, out_d), nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(out_d, out_d), nn.LayerNorm(out_d), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        return h + self.res(h)


# ---------------------------------------------------------------------------
# Evidence-aware gated fusion (used *inside* the trained specialists D-G)
# ---------------------------------------------------------------------------

class AdaptiveGatedFusion(nn.Module):
    """Gated fusion over modality embeddings, blended with an evidence prior.

    Distinct from src/fusion/adaptive_fusion.DiagnosisAdaptiveFusion: this one is
    a trained component *within* specialists D-G, whereas the latter is the
    system-level EM fusion across specialists.
    """

    def __init__(self, embed_dim: int = 256, num_classes: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.gating_net = nn.Sequential(
            nn.Linear(embed_dim + 1, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.act = nn.GELU()

    def forward(self, modality_embeddings: List[torch.Tensor],
                evidence_scores: List[torch.Tensor]):
        gate_logits = []
        for emb, ev in zip(modality_embeddings, evidence_scores):
            emb_n = F.normalize(emb, p=2, dim=-1)
            gate_logits.append(self.gating_net(torch.cat([emb_n, ev], dim=1)))

        raw_weights = F.softmax(torch.cat(gate_logits, dim=1) / 2.0, dim=1)
        ev_stacked = torch.cat(evidence_scores, dim=1)
        ev_prior = ev_stacked / (torch.sum(ev_stacked, dim=1, keepdim=True) + 1e-7)
        dynamic_weights = 0.60 * raw_weights + 0.40 * ev_prior
        dynamic_weights = dynamic_weights / torch.sum(dynamic_weights, dim=1, keepdim=True)

        fused = torch.zeros_like(modality_embeddings[0])
        for i, emb in enumerate(modality_embeddings):
            fused = fused + dynamic_weights[:, i:i + 1] * emb

        fused_rep = self.act(fused + self.fusion_projection(fused))
        return self.classifier(fused_rep), dynamic_weights, fused_rep


# ---------------------------------------------------------------------------
# Specialists A-G
# ---------------------------------------------------------------------------

def _head(embed_dim: int, num_classes: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(embed_dim, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(128, num_classes),
    )


class ModelA(nn.Module):
    """Image only."""

    def __init__(self, embed_dim=256, num_classes=6, pretrained=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.classifier = _head(embed_dim, num_classes)

    def forward(self, img=None, img_embed=None):
        if img_embed is None and img is not None:
            img_embed = self.image_encoder(img)
        elif img_embed is not None and img_embed.shape[-1] in (2048, 1792):
            img_embed = self.image_encoder(img_embed)
        return self.classifier(img_embed), img_embed


class ModelB(nn.Module):
    """Metadata only."""

    def __init__(self, meta_input_dim=116, embed_dim=256, num_classes=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.meta_encoder = MetadataEncoder(input_dim=meta_input_dim, embed_dim=embed_dim)
        self.classifier = _head(embed_dim, num_classes)

    def forward(self, meta=None, meta_embed=None):
        if meta_embed is None and meta is not None:
            meta_embed = self.meta_encoder(meta)
        elif meta_embed is not None and meta_embed.shape[-1] != self.embed_dim:
            meta_embed = self.meta_encoder(meta_embed)
        return self.classifier(meta_embed), meta_embed


class ModelC(nn.Module):
    """Free text only."""

    def __init__(self, embed_dim=256, num_classes=6, load_bert="config"):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_encoder = LanguageEncoder(embed_dim=embed_dim, load_bert=load_bert)
        self.res_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(0.15),
        )
        self.classifier = _head(embed_dim, num_classes, dropout=0.1)

    def forward(self, input_ids=None, attention_mask=None, text_embed=None):
        if text_embed is None and input_ids is not None:
            text_embed = self.text_encoder(input_ids, attention_mask)
        elif text_embed is not None and text_embed.shape[-1] != self.embed_dim:
            text_embed = self.text_encoder.projection(text_embed)
        h = text_embed + self.res_block(text_embed)
        return self.classifier(h), h


class _FusedSpecialist(nn.Module):
    """Shared plumbing for the gated-fusion specialists D-G."""

    DEFAULT_EVIDENCE = {"img": 0.85, "meta": 0.80, "text": 0.85}

    def _evidence(self, value: Optional[torch.Tensor], key: str,
                  batch: int, device) -> torch.Tensor:
        if value is not None:
            return value
        return torch.ones(batch, 1, device=device) * self.DEFAULT_EVIDENCE[key]


class ModelD(_FusedSpecialist):
    """Image + metadata."""

    def __init__(self, meta_input_dim=116, embed_dim=256, num_classes=6, pretrained=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.meta_encoder = MetadataEncoder(input_dim=meta_input_dim, embed_dim=embed_dim)
        self.fusion = AdaptiveGatedFusion(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, img=None, meta=None, img_embed=None, meta_embed=None,
                ev_img=None, ev_meta=None):
        if img_embed is None and img is not None:
            img_embed = self.image_encoder(img)
        elif img_embed is not None and img_embed.shape[-1] != self.embed_dim:
            img_embed = self.image_encoder.projection(img_embed)
        if meta_embed is None and meta is not None:
            meta_embed = self.meta_encoder(meta)
        elif meta_embed is not None and meta_embed.shape[-1] != self.embed_dim:
            meta_embed = self.meta_encoder(meta_embed)

        b, dev = img_embed.size(0), img_embed.device
        return self.fusion([img_embed, meta_embed],
                           [self._evidence(ev_img, "img", b, dev),
                            self._evidence(ev_meta, "meta", b, dev)])


class ModelE(_FusedSpecialist):
    """Image + free text."""

    def __init__(self, embed_dim=256, num_classes=6, pretrained=False, load_bert="config"):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.text_encoder = LanguageEncoder(embed_dim=embed_dim, load_bert=load_bert)
        self.fusion = AdaptiveGatedFusion(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, img=None, input_ids=None, attention_mask=None,
                img_embed=None, text_embed=None, ev_img=None, ev_text=None):
        if img_embed is None and img is not None:
            img_embed = self.image_encoder(img)
        elif img_embed is not None and img_embed.shape[-1] != self.embed_dim:
            img_embed = self.image_encoder.projection(img_embed)
        if text_embed is None and input_ids is not None:
            text_embed = self.text_encoder(input_ids, attention_mask)
        elif text_embed is not None and text_embed.shape[-1] != self.embed_dim:
            text_embed = self.text_encoder.projection(text_embed)

        b, dev = img_embed.size(0), img_embed.device
        return self.fusion([img_embed, text_embed],
                           [self._evidence(ev_img, "img", b, dev),
                            self._evidence(ev_text, "text", b, dev)])


class ModelF(_FusedSpecialist):
    """Free text + metadata. Fusion order is [text, metadata] - matches training."""

    def __init__(self, meta_input_dim=116, embed_dim=256, num_classes=6, load_bert="config"):
        super().__init__()
        self.embed_dim = embed_dim
        self.meta_encoder = MetadataEncoder(input_dim=meta_input_dim, embed_dim=embed_dim)
        self.text_encoder = LanguageEncoder(embed_dim=embed_dim, load_bert=load_bert)
        self.fusion = AdaptiveGatedFusion(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, meta=None, input_ids=None, attention_mask=None,
                meta_embed=None, text_embed=None, ev_meta=None, ev_text=None):
        if meta_embed is None and meta is not None:
            meta_embed = self.meta_encoder(meta)
        elif meta_embed is not None and meta_embed.shape[-1] != self.embed_dim:
            meta_embed = self.meta_encoder(meta_embed)
        if text_embed is None and input_ids is not None:
            text_embed = self.text_encoder(input_ids, attention_mask)
        elif text_embed is not None and text_embed.shape[-1] != self.embed_dim:
            text_embed = self.text_encoder.projection(text_embed)

        b, dev = meta_embed.size(0), meta_embed.device
        return self.fusion([text_embed, meta_embed],
                           [self._evidence(ev_text, "text", b, dev),
                            self._evidence(ev_meta, "meta", b, dev)])


class ModelG(_FusedSpecialist):
    """Full tri-modal: image + metadata + free text."""

    def __init__(self, meta_input_dim=116, embed_dim=256, num_classes=6,
                 pretrained=False, load_bert="config"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.meta_encoder = MetadataEncoder(input_dim=meta_input_dim, embed_dim=embed_dim)
        self.text_encoder = LanguageEncoder(embed_dim=embed_dim, load_bert=load_bert)
        self.fusion = AdaptiveGatedFusion(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, img=None, meta=None, input_ids=None, attention_mask=None,
                img_embed=None, meta_embed=None, text_embed=None,
                ev_img=None, ev_meta=None, ev_text=None):
        if img_embed is None and img is not None:
            img_embed = self.image_encoder(img)
        elif img_embed is not None and img_embed.shape[-1] != self.embed_dim:
            img_embed = self.image_encoder.projection(img_embed)
        if meta_embed is None and meta is not None:
            meta_embed = self.meta_encoder(meta)
        elif meta_embed is not None and meta_embed.shape[-1] != self.embed_dim:
            meta_embed = self.meta_encoder(meta_embed)
        if text_embed is None and input_ids is not None:
            text_embed = self.text_encoder(input_ids, attention_mask)
        elif text_embed is not None and text_embed.shape[-1] != self.embed_dim:
            text_embed = self.text_encoder.projection(text_embed)

        b = img_embed.size(0) if img_embed is not None else meta_embed.size(0)
        dev = img_embed.device if img_embed is not None else meta_embed.device
        return self.fusion([img_embed, meta_embed, text_embed],
                           [self._evidence(ev_img, "img", b, dev),
                            self._evidence(ev_meta, "meta", b, dev),
                            self._evidence(ev_text, "text", b, dev)])


SPECIALIST_CLASSES = {"A": ModelA, "B": ModelB, "C": ModelC,
                      "D": ModelD, "E": ModelE, "F": ModelF, "G": ModelG}

# Which modalities each specialist consumes - drives routing in backends.py.
SPECIALIST_MODALITIES = {
    "A": ("image",), "B": ("metadata",), "C": ("text",),
    "D": ("image", "metadata"), "E": ("image", "text"),
    "F": ("metadata", "text"), "G": ("image", "metadata", "text"),
}
