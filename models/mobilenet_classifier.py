"""
MobileNetV3 Classifier for Edge Computing Video Analytics
==========================================================
Author: Pranay M Mahendrakar
Description: Lightweight MobileNetV3-based image classifier optimized for
             real-time inference on edge devices. Uses depthwise separable
             convolutions and SE blocks for efficiency.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from models.base_model import BaseModel, InferenceResult, ModelConfig

logger = logging.getLogger(__name__)

# ImageNet 1000 class count
IMAGENET_NUM_CLASSES = 1000

# MobileNetV3 variant parameters
MOBILENET_CONFIGS = {
    "small": {
        "model_fn": models.mobilenet_v3_small,
        "weights": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "params": "2.5M",
        "top1_acc": 67.668,
        "input_size": (224, 224),
    },
    "large": {
        "model_fn": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "params": "5.4M",
        "top1_acc": 75.274,
        "input_size": (224, 224),
    },
}


class MobileNetClassifier(BaseModel):
    """
    MobileNetV3-based lightweight image classifier for edge deployment.

    MobileNetV3 uses depthwise separable convolutions and hardware-aware NAS
    to achieve excellent accuracy/efficiency tradeoffs on mobile and edge devices.

    Architecture highlights:
    - Depthwise separable convolutions (9x fewer FLOPs vs standard conv)
    - Squeeze-and-Excitation (SE) attention blocks
    - Hard-Swish activation for better performance without exp() overhead
    - Neural Architecture Search (NAS) optimized structure

    Args:
        variant: 'small' (2.5M params) or 'large' (5.4M params)
        num_classes: Number of output classes (1000 for ImageNet pretrained)
        config: ModelConfig with device and inference settings
        custom_class_names: Custom class label names for output

    Example:
        classifier = MobileNetClassifier(variant='small')
        classifier.load_model()
        classifier.warmup()

        frame = cv2.imread('scene.jpg')
        result = classifier.predict(frame)
        print(f"Top prediction: {result.classifications[0]['class_name']}")
        print(f"Confidence: {result.classifications[0]['confidence']:.3f}")
        print(f"Latency: {result.total_latency_ms:.1f}ms")
    """

    VALID_VARIANTS = list(MOBILENET_CONFIGS.keys())

    def __init__(
        self,
        variant: str = "small",
        num_classes: int = IMAGENET_NUM_CLASSES,
        config: Optional[ModelConfig] = None,
        custom_class_names: Optional[List[str]] = None,
        top_k: int = 5,
    ):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"variant must be one of {self.VALID_VARIANTS}")

        self.variant = variant
        self.top_k = top_k
        self._variant_config = MOBILENET_CONFIGS[variant]
        input_size = self._variant_config["input_size"]

        if config is None:
            config = ModelConfig(
                model_name=f"mobilenet_v3_{variant}",
                input_size=input_size,
                num_classes=num_classes,
                class_names=custom_class_names,
            )
        else:
            config.model_name = f"mobilenet_v3_{variant}"
            config.input_size = input_size
            config.num_classes = num_classes
            if custom_class_names:
                config.class_names = custom_class_names

        super().__init__(config)

        # Build preprocessing transform pipeline
        self._transform = self._build_transform()
        logger.info(
            f"MobileNetClassifier initialized: v3-{variant} | "
            f"params={self._variant_config['params']} | "
            f"top-1={self._variant_config['top1_acc']}%"
        )

    def _build_transform(self) -> transforms.Compose:
        """Build the standard ImageNet preprocessing transform pipeline."""
        h, w = self.config.input_size
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def load_model(
        self,
        model_path: Optional[str] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """
        Load MobileNetV3 model weights.

        Args:
            model_path: Path to custom weights (.pth) file. If None, uses pretrained ImageNet.
            pretrained: Whether to load ImageNet pretrained weights (if no custom path)
            freeze_backbone: Freeze backbone for feature extraction only
        """
        logger.info(f"Loading MobileNetV3-{self.variant}...")
        t_start = time.perf_counter()

        variant_cfg = self._variant_config

        if model_path is not None:
            # Load custom weights (no pretrained)
            model_fn = variant_cfg["model_fn"]
            self.model = model_fn(weights=None)

            # Adjust final layer if needed
            if self.config.num_classes != IMAGENET_NUM_CLASSES:
                self._replace_head(self.config.num_classes)

            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Custom weights loaded from: {model_path}")
        else:
            if pretrained:
                self.model = variant_cfg["model_fn"](weights=variant_cfg["weights"])
                logger.info("Pretrained ImageNet weights loaded")
            else:
                self.model = variant_cfg["model_fn"](weights=None)
                logger.warning("Randomly initialized weights (no pretrained)")

            if self.config.num_classes != IMAGENET_NUM_CLASSES:
                self._replace_head(self.config.num_classes)

        # Freeze backbone for faster fine-tuning
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen for feature extraction mode")

        # Optimize for inference
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.config.half_precision and str(self.device) == "cuda":
            self.model = self.model.half()
            logger.info("FP16 half precision enabled")

        self._is_loaded = True
        load_time = (time.perf_counter() - t_start) * 1000

        logger.info(
            f"Model loaded in {load_time:.0f}ms | "
            f"Device: {self.device} | "
            f"Input: {self.config.input_size}"
        )

    def _replace_head(self, num_classes: int) -> None:
        """Replace the classification head for custom number of classes."""
        # MobileNetV3 classifier head structure
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        logger.info(f"Classification head replaced: {in_features} → {num_classes} classes")

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess BGR frame for MobileNetV3 input.

        Pipeline: BGR → RGB → Resize(224x224) → Normalize → Tensor → Batch

        Args:
            frame: Input BGR image as numpy array (H, W, C)

        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transform pipeline
        tensor = self._transform(rgb_frame)

        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)

        if self.config.half_precision and str(self.device) == "cuda":
            tensor = tensor.half()

        return tensor

    def infer(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run MobileNetV3 forward pass.

        Args:
            preprocessed_input: Preprocessed tensor (1, 3, H, W)

        Returns:
            Raw logits tensor (1, num_classes)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with torch.no_grad():
            logits = self.model(preprocessed_input)

        return logits

    def postprocess(
        self,
        raw_output: torch.Tensor,
        original_shape: Tuple[int, int],
    ) -> InferenceResult:
        """
        Convert raw logits to top-k classification results.

        Args:
            raw_output: Logits tensor (1, num_classes)
            original_shape: Original frame shape (unused for classification)

        Returns:
            InferenceResult with top-k classifications
        """
        result = InferenceResult()

        # Convert logits to probabilities
        probabilities = F.softmax(raw_output, dim=1).squeeze()
        probabilities_np = probabilities.cpu().float().numpy()

        # Get top-k predictions
        k = min(self.top_k, self.config.num_classes)
        top_k_indices = np.argsort(probabilities_np)[::-1][:k]

        classifications = []
        for rank, class_id in enumerate(top_k_indices):
            confidence = float(probabilities_np[class_id])
            class_name = (
                self.config.class_names[class_id]
                if self.config.class_names and class_id < len(self.config.class_names)
                else f"class_{class_id}"
            )
            classifications.append({
                "rank": rank + 1,
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": round(confidence, 6),
                "confidence_pct": round(confidence * 100, 2),
            })

        result.classifications = classifications
        result.metadata = {
            "model": f"mobilenet_v3_{self.variant}",
            "top_k": k,
            "num_classes": self.config.num_classes,
        }
        return result

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract feature embeddings for downstream tasks (e.g., re-ID, clustering).

        Args:
            frame: Input BGR frame

        Returns:
            Feature vector as numpy array
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded.")

        tensor = self.preprocess(frame)

        with torch.no_grad():
            # Use features (backbone) without classification head
            features = self.model.features(tensor)
            features = self.model.avgpool(features)
            features = torch.flatten(features, 1)

        return features.cpu().float().numpy().squeeze()

    def draw_classification(
        self,
        frame: np.ndarray,
        result: InferenceResult,
        show_top: int = 3,
        show_fps: bool = True,
    ) -> np.ndarray:
        """
        Overlay classification results on frame.

        Args:
            frame: Input BGR frame
            result: InferenceResult from predict()
            show_top: Number of top predictions to display
            show_fps: Whether to show FPS overlay

        Returns:
            Annotated frame with classification labels
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (350, 30 + show_top * 30 + 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # Model label
        cv2.putText(
            annotated, f"MobileNetV3-{self.variant.capitalize()}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        # Top-k classifications
        for i, cls in enumerate(result.classifications[:show_top]):
            y_pos = 60 + i * 30
            bar_width = int(cls["confidence"] * 300)
            color = (0, 255, 0) if i == 0 else (0, 200, 200)

            # Confidence bar
            cv2.rectangle(annotated, (15, y_pos - 15), (15 + bar_width, y_pos), color, -1)

            # Label
            label = f"{cls['rank']}. {cls['class_name']}: {cls['confidence_pct']:.1f}%"
            cv2.putText(
                annotated, label, (20, y_pos - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )

        # FPS overlay
        if show_fps and result.fps > 0:
            fps_text = f"FPS: {result.fps:.1f} | {result.total_latency_ms:.1f}ms"
            cv2.putText(
                annotated, fps_text, (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        return annotated


# ─── Convenience Factory ─────────────────────────────────────────────────────

def create_classifier(
    variant: str = "small",
    device: str = "cpu",
    num_classes: int = 1000,
    top_k: int = 5,
    class_names: Optional[List[str]] = None,
) -> MobileNetClassifier:
    """
    Factory function to create a configured MobileNetClassifier.

    Args:
        variant: 'small' or 'large'
        device: 'cpu' | 'cuda'
        num_classes: Number of output classes
        top_k: Number of top predictions to return
        class_names: Optional list of class label strings

    Returns:
        Configured MobileNetClassifier ready for load_model()
    """
    config = ModelConfig(
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        half_precision=(device == "cuda"),
    )
    return MobileNetClassifier(variant=variant, config=config, top_k=top_k)
