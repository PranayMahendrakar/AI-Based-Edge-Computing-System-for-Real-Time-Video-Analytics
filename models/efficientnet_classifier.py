"""
EfficientNet Classifier for Edge Computing Video Analytics
==========================================================
Author: Pranay M Mahendrakar
Description: EfficientNet-based image classifier with compound scaling for
             optimal accuracy/efficiency tradeoffs on edge devices.
             Supports B0-B3 variants via timm library.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models.base_model import BaseModel, InferenceResult, ModelConfig

logger = logging.getLogger(__name__)

# EfficientNet variant specifications
EFFICIENTNET_CONFIGS = {
    "b0": {
        "timm_name": "efficientnet_b0",
        "params": "5.3M",
        "top1_acc": 77.1,
        "input_size": (224, 224),
        "flops": "0.39B",
    },
    "b1": {
        "timm_name": "efficientnet_b1",
        "params": "7.8M",
        "top1_acc": 79.1,
        "input_size": (240, 240),
        "flops": "0.70B",
    },
    "b2": {
        "timm_name": "efficientnet_b2",
        "params": "9.2M",
        "top1_acc": 80.1,
        "input_size": (260, 260),
        "flops": "1.00B",
    },
    "b3": {
        "timm_name": "efficientnet_b3",
        "params": "12.2M",
        "top1_acc": 81.6,
        "input_size": (300, 300),
        "flops": "1.83B",
    },
    # Lite variants optimized for mobile/edge
    "lite0": {
        "timm_name": "efficientnet_lite0",
        "params": "4.7M",
        "top1_acc": 75.5,
        "input_size": (224, 224),
        "flops": "0.40B",
    },
}

# Recommended variants for edge deployment
EDGE_RECOMMENDED = {
    "ultra_fast": "b0",     # Best speed, minimal memory
    "balanced": "b1",       # Good accuracy/speed balance
    "accurate": "b2",       # Higher accuracy, moderate speed
    "edge_optimized": "lite0",  # Specifically optimized for edge
}


class EfficientNetClassifier(BaseModel):
    """
    EfficientNet-based image classifier using compound scaling.

    EfficientNet uses Neural Architecture Search (NAS) and a compound scaling
    method that uniformly scales network width, depth, and resolution, achieving
    state-of-the-art accuracy with significantly fewer parameters than alternatives.

    Compound Scaling Formula:
        depth:    d = alpha^phi
        width:    w = beta^phi
        resolution: r = gamma^phi
        where alpha * beta^2 * gamma^2 ≈ 2 (resource constraint)

    Args:
        variant: EfficientNet variant ('b0' through 'b3', or 'lite0')
        num_classes: Number of output classes
        config: ModelConfig instance
        custom_class_names: Optional class label names
        top_k: Number of top-k predictions to return
        use_timm: Use timm library (True) or torchvision (False)

    Example:
        # Create and load B0 (fastest, edge-optimized)
        classifier = EfficientNetClassifier(variant='b0', use_timm=True)
        classifier.load_model()
        classifier.warmup()

        frame = cv2.imread('image.jpg')
        result = classifier.predict(frame)
        top1 = result.classifications[0]
        print(f"{top1['class_name']}: {top1['confidence_pct']:.1f}% ({result.fps:.1f} FPS)")
    """

    VALID_VARIANTS = list(EFFICIENTNET_CONFIGS.keys())

    def __init__(
        self,
        variant: str = "b0",
        num_classes: int = 1000,
        config: Optional[ModelConfig] = None,
        custom_class_names: Optional[List[str]] = None,
        top_k: int = 5,
        use_timm: bool = True,
    ):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {self.VALID_VARIANTS}. "
                f"Recommended for edge: {list(EDGE_RECOMMENDED.values())}"
            )

        self.variant = variant
        self.top_k = top_k
        self.use_timm = use_timm
        self._variant_config = EFFICIENTNET_CONFIGS[variant]
        input_size = self._variant_config["input_size"]

        if config is None:
            config = ModelConfig(
                model_name=f"efficientnet_{variant}",
                input_size=input_size,
                num_classes=num_classes,
                class_names=custom_class_names,
            )
        else:
            config.model_name = f"efficientnet_{variant}"
            config.input_size = input_size
            config.num_classes = num_classes
            if custom_class_names:
                config.class_names = custom_class_names

        super().__init__(config)
        self._transform = self._build_transform()

        logger.info(
            f"EfficientNetClassifier initialized: {variant.upper()} | "
            f"params={self._variant_config['params']} | "
            f"top-1={self._variant_config['top1_acc']}% | "
            f"FLOPs={self._variant_config['flops']}"
        )

    def _build_transform(self) -> transforms.Compose:
        """Build preprocessing transform matching EfficientNet training pipeline."""
        h, w = self.config.input_size
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(h * 1.14), int(w * 1.14))),  # Slightly larger for crop
            transforms.CenterCrop((h, w)),
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
    ) -> None:
        """
        Load EfficientNet model weights.

        Args:
            model_path: Custom weights file path (.pth). If None, uses pretrained ImageNet.
            pretrained: Load ImageNet pretrained weights (if no custom path)
        """
        logger.info(f"Loading EfficientNet-{self.variant.upper()}...")
        t_start = time.perf_counter()

        if self.use_timm:
            self._load_with_timm(model_path, pretrained)
        else:
            self._load_with_torchvision(model_path, pretrained)

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.config.half_precision and str(self.device) == "cuda":
            self.model = self.model.half()
            logger.info("FP16 half precision enabled")

        self._is_loaded = True
        load_time = (time.perf_counter() - t_start) * 1000

        # Count parameters
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(
            f"Model loaded in {load_time:.0f}ms | "
            f"Params: {param_count:.1f}M | "
            f"Input: {self.config.input_size} | "
            f"Device: {self.device}"
        )

    def _load_with_timm(self, model_path: Optional[str], pretrained: bool) -> None:
        """Load using timm library for full EfficientNet variant support."""
        try:
            import timm
        except ImportError:
            raise ImportError("timm package required. Install with: pip install timm")

        timm_name = self._variant_config["timm_name"]

        if model_path is not None:
            self.model = timm.create_model(
                timm_name,
                pretrained=False,
                num_classes=self.config.num_classes
            )
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Custom weights loaded: {model_path}")
        else:
            self.model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=self.config.num_classes
            )
            if pretrained:
                logger.info(f"Pretrained ImageNet weights loaded via timm")

    def _load_with_torchvision(self, model_path: Optional[str], pretrained: bool) -> None:
        """Load using torchvision (limited to B0-B7 variants)."""
        from torchvision import models as tv_models
        import torchvision

        variant_map = {
            "b0": (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            "b1": (tv_models.efficientnet_b1, tv_models.EfficientNet_B1_Weights.IMAGENET1K_V1),
            "b2": (tv_models.efficientnet_b2, tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1),
            "b3": (tv_models.efficientnet_b3, tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1),
        }

        if self.variant not in variant_map:
            raise ValueError(f"torchvision does not support variant '{self.variant}'. Use timm.")

        model_fn, weights = variant_map[self.variant]

        if model_path:
            self.model = model_fn(weights=None)
            if self.config.num_classes != 1000:
                in_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features, self.config.num_classes)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = model_fn(weights=weights if pretrained else None)
            if self.config.num_classes != 1000:
                in_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features, self.config.num_classes)

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess BGR frame for EfficientNet input.

        Pipeline: BGR → RGB → Resize+CenterCrop → Normalize → Tensor → Batch

        Args:
            frame: Input BGR frame (H, W, C)

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb_frame)
        tensor = tensor.unsqueeze(0).to(self.device)

        if self.config.half_precision and str(self.device) == "cuda":
            tensor = tensor.half()

        return tensor

    def infer(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run EfficientNet forward pass.

        Args:
            preprocessed_input: Preprocessed tensor (1, 3, H, W)

        Returns:
            Logits tensor (1, num_classes)
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
        Convert raw logits to structured top-k classification results.

        Args:
            raw_output: Logits tensor (1, num_classes)
            original_shape: Original frame shape (unused for classification)

        Returns:
            InferenceResult with top-k classifications sorted by confidence
        """
        result = InferenceResult()

        # Convert to probabilities
        probabilities = F.softmax(raw_output, dim=1).squeeze()
        prob_np = probabilities.cpu().float().numpy()

        # Get top-k predictions
        k = min(self.top_k, self.config.num_classes)
        top_k_indices = np.argsort(prob_np)[::-1][:k]

        classifications = []
        for rank, class_id in enumerate(top_k_indices):
            confidence = float(prob_np[class_id])
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
            "model": f"efficientnet_{self.variant}",
            "top_k": k,
            "num_classes": self.config.num_classes,
            "flops": self._variant_config["flops"],
        }
        return result

    def get_compound_scaling_info(self) -> Dict[str, Any]:
        """Return compound scaling coefficients for this variant."""
        # EfficientNet compound scaling coefficients (phi=1 for B0 baseline)
        scaling_table = {
            "b0": {"phi": 0, "resolution": 224, "depth": 1.0, "width": 1.0},
            "b1": {"phi": 0.5, "resolution": 240, "depth": 1.1, "width": 1.0},
            "b2": {"phi": 1, "resolution": 260, "depth": 1.2, "width": 1.1},
            "b3": {"phi": 2, "resolution": 300, "depth": 1.4, "width": 1.2},
        }
        return scaling_table.get(self.variant, {})

    def draw_classification(
        self,
        frame: np.ndarray,
        result: InferenceResult,
        show_top: int = 3,
        show_fps: bool = True,
    ) -> np.ndarray:
        """Overlay EfficientNet classification results on frame."""
        annotated = frame.copy()

        # Semi-transparent panel background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (380, 30 + show_top * 30 + 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        # Header
        cv2.putText(
            annotated, f"EfficientNet-{self.variant.upper()}",
            (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2
        )

        # Top-k results
        for i, cls in enumerate(result.classifications[:show_top]):
            y = 65 + i * 30
            conf = cls["confidence"]
            bar_w = int(conf * 320)
            color = (0, 255, 0) if i == 0 else (0, 200, 200)

            # Confidence bar
            cv2.rectangle(annotated, (15, y - 14), (15 + bar_w, y + 2), color, -1)

            # Label text
            label = f"#{cls['rank']} {cls['class_name']}: {cls['confidence_pct']:.1f}%"
            cv2.putText(
                annotated, label, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )

        # FPS overlay
        if show_fps and result.fps > 0:
            h, w = frame.shape[:2]
            fps_text = f"FPS: {result.fps:.1f} | {result.total_latency_ms:.1f}ms"
            cv2.putText(
                annotated, fps_text, (w - 210, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        return annotated


# ─── Convenience Factory ─────────────────────────────────────────────────────

def create_efficientnet(
    variant: str = "b0",
    device: str = "cpu",
    num_classes: int = 1000,
    top_k: int = 5,
    class_names: Optional[List[str]] = None,
    use_timm: bool = True,
) -> EfficientNetClassifier:
    """
    Factory function to create a configured EfficientNetClassifier.

    Args:
        variant: 'b0' | 'b1' | 'b2' | 'b3' | 'lite0'
        device: 'cpu' | 'cuda'
        num_classes: Number of output classes
        top_k: Number of top-k predictions to return
        class_names: Optional class label strings
        use_timm: Use timm for loading (recommended for full variant support)

    Returns:
        Configured EfficientNetClassifier ready for load_model()
    """
    config = ModelConfig(
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        half_precision=(device == "cuda"),
    )
    return EfficientNetClassifier(
        variant=variant, config=config, top_k=top_k, use_timm=use_timm
    )
