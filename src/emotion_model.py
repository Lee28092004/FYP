# src/emotion_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


@dataclass
class EmotionPrediction:
    label: str
    confidence: float
    probs: List[float]


def build_mobilenetv2(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class EmotionRecognizer:
    def __init__(
        self,
        weights_path: str,
        class_names: List[str],
        device: str = "cpu",
    ) -> None:
        self.class_names = class_names
        self.device = torch.device(device)

        self.model = build_mobilenetv2(num_classes=len(class_names))
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def preprocess_face_bgr(face_bgr: np.ndarray, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Input: BGR face image (OpenCV)
        Output: torch tensor [1,3,224,224] normalized like ImageNet
        """
        face = cv2.resize(face_bgr, size, interpolation=cv2.INTER_AREA)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        face_rgb = (face_rgb - mean) / std

        # HWC -> CHW
        chw = np.transpose(face_rgb, (2, 0, 1))
        x = torch.from_numpy(chw).unsqueeze(0)  # [1,3,H,W]
        return x

    @torch.no_grad()
    def predict(self, face_bgr: np.ndarray) -> EmotionPrediction:
        x = self.preprocess_face_bgr(face_bgr).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return EmotionPrediction(
            label=self.class_names[idx],
            confidence=float(probs[idx]),
            probs=probs.tolist(),
        )
