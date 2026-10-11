"""
src/specialists - the Vision Specialist Team (from the original DERMA-GUARD design).

The original system ran several independently trained models on each case and let
an evidence manager decide whom to trust. The final system keeps that idea with
models that are actually trained and verified:

    S1  MILK10k clinical-photo head      evidential, ResNet50 fine-tuned on MILK10k
    S2  MILK10k dermoscopy head          evidential, same backbone
    S3  MILK10k metadata head            evidential, residual MLP
    S4  PAD-UFES-20 Model A (legacy)     the ORIGINAL project's image specialist -
                                         ResNet50 + channel attention + MHSA ViT,
                                         trained on smartphone photos (PAD-UFES-20),
                                         run here on MILK10k clinical photos
    S5  Case-based (kNN) specialist      votes of the most similar TRAINING lesions

S4 is an external model: it never saw MILK10k, so its accuracy here is a genuine
out-of-dataset measurement. Its 6 PAD-UFES classes map onto 6 of MILK10k's 11:

    BCC->BCC  ACK->AKIEC  NEV->NV  SEK->BKL  SCC->SCCKA  MEL->MEL

and it puts no probability on the other 5 (BEN_OTH, DF, INF, MAL_OTH, VASC).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PAD_DIR = ROOT / "runs" / "pad_ufes_legacy" / "checkpoints"

PAD_CLASSES = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL"]
PAD_TO_MILK = {"BCC": "BCC", "ACK": "AKIEC", "NEV": "NV", "SEK": "BKL",
               "SCC": "SCCKA", "MEL": "MEL"}


class PadLegacySpecialist:
    """The original project's Model A, applied to MILK10k photos."""

    def __init__(self, milk_classes: List[str], weights_dir: Path = PAD_DIR,
                 device: str = "cpu"):
        import torchvision.models as tvm
        import torchvision.transforms as T

        from src.specialists.pad_legacy_arch import ModelA, VisionTransformerImageEncoder

        self.available = (weights_dir / "model_A.pt").exists() and \
            (weights_dir / "img_proj.pt").exists()
        if not self.available:
            return
        self.device = torch.device(device)
        self.resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = torch.nn.Identity()
        self.resnet.eval().to(self.device)
        self.proj = VisionTransformerImageEncoder(embed_dim=256, num_tokens=8, num_heads=8,
                                                  num_layers=2, pretrained=False)
        self.proj.load_state_dict(torch.load(weights_dir / "img_proj.pt", map_location="cpu",
                                             weights_only=False), strict=True)
        self.proj.eval().to(self.device)
        self.model = ModelA(num_classes=6, pretrained=False)
        self.model.load_state_dict(torch.load(weights_dir / "model_A.pt", map_location="cpu",
                                              weights_only=False), strict=True)
        self.model.eval().to(self.device)
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.cols = [milk_classes.index(PAD_TO_MILK[c]) for c in PAD_CLASSES]
        self.K = len(milk_classes)

    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image]) -> np.ndarray:
        """(N, K_milk) probabilities; zero on the 5 classes PAD-UFES does not have."""
        x = torch.stack([self.tf(im.convert("RGB")) for im in images]).to(self.device)
        logits, _ = self.model(img_embed=self.proj(self.resnet(x)))
        p6 = F.softmax(logits, dim=-1).cpu().numpy()
        out = np.zeros((len(images), self.K), dtype=np.float32)
        out[:, self.cols] = p6
        return out

    def predict(self, image: Optional[Image.Image]) -> Optional[np.ndarray]:
        if image is None or not self.available:
            return None
        return self.predict_batch([image])[0]
