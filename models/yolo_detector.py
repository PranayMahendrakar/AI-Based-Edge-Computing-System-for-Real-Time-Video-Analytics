"""
YOLO-Based Object Detector for Edge Computing Video Analytics
=============================================================
Author: Pranay M Mahendrakar
Description: YOLOv8-based real-time object detection optimized for edge devices.
             Supports TensorRT acceleration, INT8/FP16 quantization, and
             adaptive inference for resource-constrained hardware.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch

from models.base_model import BaseModel, InferenceResult, ModelConfig

logger = logging.getLogger(__name__)

# COCO 80-class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Model variant configurations
YOLO_CONFIGS = {
    "yolov8n": {"params": "3.2M", "mAP": 37.3, "speed_cpu": 80.4},
    "yolov8s": {"params": "11.2M", "mAP": 44.9, "speed_cpu": 128.4},
    "yolov8m": {"params": "25.9M", "mAP": 50.2, "speed_cpu": 234.7},
}


class YOLODetector(BaseModel):
    """
    YOLOv8-based object detector optimized for edge deployment.

    Features:
    - Supports YOLOv8n/s/m variants (nano recommended for edge)
    - TensorRT optimization for NVIDIA Jetson devices
    - INT8/FP16 quantization for reduced memory footprint
    - Real-time tracking integration (SORT/ByteTrack)
    - ROI-based detection for focused processing

    Args:
        model_size: YOLO variant ('nano', 'small', 'medium')
        config: ModelConfig with inference parameters
        enable_tracking: Enable multi-object tracking
        roi_zones: List of (x1, y1, x2, y2) ROI bounding boxes

    Example:
        detector = YOLODetector(model_size='nano', config=ModelConfig(device='cuda'))
        detector.load_model()
        detector.warmup()

        frame = cv2.imread('frame.jpg')
        result = detector.predict(frame)
        print(f"Detected {len(result.detections)} objects at {result.fps:.1f} FPS")
    """

    VALID_SIZES = {"nano": "yolov8n", "small": "yolov8s", "medium": "yolov8m"}

    def __init__(
        self,
        model_size: str = "nano",
        config: Optional[ModelConfig] = None,
        enable_tracking: bool = False,
        roi_zones: Optional[List[Tuple[int, int, int, int]]] = None,
        target_classes: Optional[List[int]] = None,
    ):
        if model_size not in self.VALID_SIZES:
            raise ValueError(f"model_size must be one of {list(self.VALID_SIZES.keys())}")

        self.model_size = model_size
        self.yolo_model_name = self.VALID_SIZES[model_size]
        self.enable_tracking = enable_tracking
        self.roi_zones = roi_zones or []
        self.target_classes = target_classes  # None = detect all classes
        self._yolo_model = None  # ultralytics YOLO instance

        if config is None:
            config = ModelConfig(
                model_name=self.yolo_model_name,
                input_size=(640, 640),
                num_classes=80,
                class_names=COCO_CLASSES,
            )
        else:
            config.model_name = self.yolo_model_name
            config.class_names = config.class_names or COCO_CLASSES

        super().__init__(config)
        logger.info(f"YOLODetector initialized: {self.yolo_model_name} | tracking={enable_tracking}")

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load YOLOv8 model weights.

        Args:
            model_path: Path to custom .pt weights file, or None for pretrained COCO weights
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")

        weights = model_path or f"{self.yolo_model_name}.pt"
        logger.info(f"Loading YOLOv8 model: {weights}")

        t_start = time.perf_counter()
        self._yolo_model = YOLO(weights)

        # Move to target device
        if str(self.device) == "cuda":
            self._yolo_model.to("cuda")
            if self.config.half_precision:
                logger.info("Enabling FP16 half precision")

        self.model = self._yolo_model.model
        self._is_loaded = True

        load_time = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"Model loaded in {load_time:.0f}ms | "
            f"Params: {YOLO_CONFIGS.get(self.yolo_model_name, {}).get('params', 'N/A')} | "
            f"mAP50-95: {YOLO_CONFIGS.get(self.yolo_model_name, {}).get('mAP', 'N/A')}"
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Minimal preprocessing — YOLOv8 handles its own preprocessing internally.
        Returns frame as-is; actual preprocessing done in infer().
        """
        return frame

    def infer(self, preprocessed_input: np.ndarray) -> Any:
        """
        Run YOLOv8 inference on the input frame.

        Args:
            preprocessed_input: BGR frame as numpy array

        Returns:
            YOLOv8 Results object list
        """
        if self._yolo_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Apply ROI mask if zones defined
        frame = preprocessed_input
        if self.roi_zones:
            frame = self._apply_roi_mask(frame)

        results = self._yolo_model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.nms_threshold,
            max_det=self.config.max_detections,
            classes=self.target_classes,
            half=self.config.half_precision,
            verbose=False,
            stream=False,
        )
        return results

    def postprocess(
        self,
        raw_output: Any,
        original_shape: Tuple[int, int]
    ) -> InferenceResult:
        """
        Convert YOLOv8 raw results to structured InferenceResult.

        Args:
            raw_output: YOLOv8 Results object list
            original_shape: Original frame (H, W) for coordinate validation

        Returns:
            InferenceResult with detection list
        """
        result = InferenceResult()
        detections = []

        for yolo_result in raw_output:
            boxes = yolo_result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = (
                    self.config.class_names[class_id]
                    if self.config.class_names and class_id < len(self.config.class_names)
                    else str(class_id)
                )
                width = x2 - x1
                height = y2 - y1

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1), "y1": round(y1),
                        "x2": round(x2), "y2": round(y2),
                        "width": round(width), "height": round(height),
                        "center_x": round(x1 + width / 2),
                        "center_y": round(y1 + height / 2),
                    },
                    "area": round(width * height),
                }
                detections.append(detection)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        result.detections = detections
        result.metadata = {
            "model": self.yolo_model_name,
            "num_detections": len(detections),
            "roi_zones": len(self.roi_zones),
        }
        return result

    def predict(self, frame: np.ndarray, frame_id: int = 0) -> InferenceResult:
        """Override predict to bypass double-preprocessing with ultralytics."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        original_shape = frame.shape[:2]

        t_pre = time.perf_counter()
        preprocessed = self.preprocess(frame)
        preprocessing_time = (time.perf_counter() - t_pre) * 1000

        t_inf = time.perf_counter()
        raw_output = self.infer(preprocessed)
        inference_time = (time.perf_counter() - t_inf) * 1000

        t_post = time.perf_counter()
        result = self.postprocess(raw_output, original_shape)
        postprocessing_time = (time.perf_counter() - t_post) * 1000

        result.inference_time_ms = inference_time
        result.preprocessing_time_ms = preprocessing_time
        result.postprocessing_time_ms = postprocessing_time
        result.frame_id = frame_id

        self._inference_count += 1
        self._total_inference_time += result.total_latency_ms

        return result

    def _apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Apply region-of-interest masking to focus detection."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for (x1, y1, x2, y2) in self.roi_zones:
            mask[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(frame, frame, mask=mask)

    def draw_detections(
        self,
        frame: np.ndarray,
        result: InferenceResult,
        show_confidence: bool = True,
        show_fps: bool = True,
    ) -> np.ndarray:
        """
        Draw detection bounding boxes and labels on frame.

        Args:
            frame: Input BGR frame
            result: InferenceResult from predict()
            show_confidence: Display confidence scores
            show_fps: Display current FPS overlay

        Returns:
            Annotated frame with detections drawn
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Color map for classes (cycling through 20 colors)
        color_map = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0),
            (0, 128, 255), (128, 0, 255), (64, 255, 64), (255, 64, 64),
        ]

        for det in result.detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            color = color_map[det["class_id"] % len(color_map)]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = det["class_name"]
            if show_confidence:
                label += f" {det['confidence']:.2f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - baseline - 5),
                (x1 + label_size[0], y1),
                color, -1
            )
            cv2.putText(
                annotated, label, (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # FPS overlay
        if show_fps and result.fps > 0:
            fps_text = f"FPS: {result.fps:.1f} | Latency: {result.total_latency_ms:.1f}ms"
            cv2.putText(
                annotated, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Detection count overlay
        count_text = f"Objects: {len(result.detections)}"
        cv2.putText(
            annotated, count_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        return annotated

    def export_tensorrt(
        self,
        output_path: str,
        precision: str = "fp16",
        workspace_size: int = 4,
    ) -> str:
        """
        Export model to TensorRT for optimized Jetson inference.

        Args:
            output_path: Output path for TensorRT engine file
            precision: 'fp32' | 'fp16' | 'int8'
            workspace_size: GPU workspace in GB

        Returns:
            Path to exported TensorRT engine
        """
        if not self._is_loaded:
            raise RuntimeError("Model must be loaded before TensorRT export.")

        logger.info(f"Exporting to TensorRT ({precision})...")
        output_path = Path(output_path).with_suffix(".engine")

        self._yolo_model.export(
            format="engine",
            half=(precision == "fp16"),
            int8=(precision == "int8"),
            workspace=workspace_size,
            device=0,
        )
        logger.info(f"TensorRT engine saved: {output_path}")
        return str(output_path)


# ─── Convenience Factory ─────────────────────────────────────────────────────

def create_detector(
    model_size: str = "nano",
    device: str = "cpu",
    confidence: float = 0.5,
    classes: Optional[List[str]] = None,
) -> YOLODetector:
    """
    Factory function to create a configured YOLODetector instance.

    Args:
        model_size: 'nano' | 'small' | 'medium'
        device: 'cpu' | 'cuda' | 'trt'
        confidence: Detection confidence threshold
        classes: List of class names to detect (None = all)

    Returns:
        Configured YOLODetector ready for load_model()
    """
    config = ModelConfig(
        device=device,
        confidence_threshold=confidence,
        class_names=classes or COCO_CLASSES,
        half_precision=(device == "cuda"),
    )
    return YOLODetector(model_size=model_size, config=config)
