"""
Base Model Abstract Class for Edge AI Video Analytics System
============================================================
Author: Pranay M Mahendrakar
Description: Abstract base class defining the interface for all deep learning
             models used in the edge computing video analytics pipeline.
"""

import abc
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for model inference results."""
    detections: List[Dict[str, Any]] = field(default_factory=list)
    classifications: List[Dict[str, Any]] = field(default_factory=list)
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    frame_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        """Total end-to-end latency in milliseconds."""
        return self.inference_time_ms + self.preprocessing_time_ms + self.postprocessing_time_ms

    @property
    def fps(self) -> float:
        """Frames per second based on total latency."""
        if self.total_latency_ms > 0:
            return 1000.0 / self.total_latency_ms
        return 0.0


@dataclass
class ModelConfig:
    """Configuration for model initialization and inference."""
    model_name: str = "base"
    device: str = "cpu"              # cpu | cuda | trt | coral
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    max_detections: int = 100
    half_precision: bool = False     # FP16 for faster edge inference
    quantized: bool = False          # INT8 quantization
    batch_size: int = 1
    num_classes: int = 80
    class_names: Optional[List[str]] = None
    model_path: Optional[str] = None
    warmup_iterations: int = 3


class BaseModel(abc.ABC):
    """
    Abstract base class for all deep learning models in the edge analytics pipeline.
    
    All model implementations (YOLO, MobileNet, EfficientNet) must inherit from
    this class and implement the abstract methods.
    
    Usage:
        class MyModel(BaseModel):
            def load_model(self): ...
            def preprocess(self, frame): ...
            def infer(self, preprocessed): ...
            def postprocess(self, raw_output): ...
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.device = self._setup_device(config.device)
        self._is_loaded = False
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._warmup_done = False

        logger.info(f"Initializing {self.__class__.__name__} on device: {self.device}")

    def _setup_device(self, device_str: str) -> torch.device:
        """Configure the compute device for inference."""
        if device_str == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                logger.warning("CUDA not available, falling back to CPU")
                device = torch.device("cpu")
        elif device_str == "trt":
            # TensorRT - handled by subclass
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        return device

    @abc.abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load model weights and prepare for inference.
        
        Args:
            model_path: Path to model weights file. If None, downloads pretrained.
        """
        pass

    @abc.abstractmethod
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess raw frame for model input.
        
        Args:
            frame: Raw BGR image as numpy array (H, W, C)
            
        Returns:
            Preprocessed tensor ready for inference
        """
        pass

    @abc.abstractmethod
    def infer(self, preprocessed_input: torch.Tensor) -> Any:
        """
        Run forward pass inference.
        
        Args:
            preprocessed_input: Preprocessed input tensor
            
        Returns:
            Raw model output
        """
        pass

    @abc.abstractmethod
    def postprocess(self, raw_output: Any, original_shape: Tuple[int, int]) -> InferenceResult:
        """
        Post-process raw model output to structured results.
        
        Args:
            raw_output: Raw model output
            original_shape: Original frame shape (H, W) for coordinate scaling
            
        Returns:
            Structured InferenceResult
        """
        pass

    def predict(self, frame: np.ndarray, frame_id: int = 0) -> InferenceResult:
        """
        End-to-end inference pipeline: preprocess → infer → postprocess.
        
        Args:
            frame: Input BGR frame as numpy array
            frame_id: Frame identifier for tracking
            
        Returns:
            Complete InferenceResult with timing metrics
        """
        if not self._is_loaded:
            raise RuntimeError(f"Model not loaded. Call load_model() first.")

        original_shape = frame.shape[:2]

        # Preprocessing
        t_pre = time.perf_counter()
        preprocessed = self.preprocess(frame)
        preprocessing_time = (time.perf_counter() - t_pre) * 1000

        # Inference
        t_inf = time.perf_counter()
        with torch.no_grad():
            raw_output = self.infer(preprocessed)
        inference_time = (time.perf_counter() - t_inf) * 1000

        # Postprocessing
        t_post = time.perf_counter()
        result = self.postprocess(raw_output, original_shape)
        postprocessing_time = (time.perf_counter() - t_post) * 1000

        # Update timing stats
        result.inference_time_ms = inference_time
        result.preprocessing_time_ms = preprocessing_time
        result.postprocessing_time_ms = postprocessing_time
        result.frame_id = frame_id

        # Update running stats
        self._inference_count += 1
        self._total_inference_time += result.total_latency_ms

        logger.debug(
            f"Frame {frame_id}: inference={inference_time:.1f}ms, "
            f"total={result.total_latency_ms:.1f}ms, fps={result.fps:.1f}"
        )

        return result

    def warmup(self) -> None:
        """Run warmup iterations to initialize GPU/hardware caches."""
        if self._warmup_done:
            return

        logger.info(f"Running {self.config.warmup_iterations} warmup iterations...")
        dummy_frame = np.zeros(
            (*self.config.input_size, 3), dtype=np.uint8
        )

        for i in range(self.config.warmup_iterations):
            self.predict(dummy_frame, frame_id=-1)

        self._warmup_done = True
        logger.info("Warmup complete.")

    def get_performance_stats(self) -> Dict[str, float]:
        """Return aggregated performance statistics."""
        if self._inference_count == 0:
            return {"avg_latency_ms": 0.0, "avg_fps": 0.0, "total_frames": 0}

        avg_latency = self._total_inference_time / self._inference_count
        return {
            "avg_latency_ms": round(avg_latency, 2),
            "avg_fps": round(1000.0 / avg_latency if avg_latency > 0 else 0, 1),
            "total_frames": self._inference_count,
            "total_time_s": round(self._total_inference_time / 1000, 2),
        }

    def reset_stats(self) -> None:
        """Reset performance tracking counters."""
        self._inference_count = 0
        self._total_inference_time = 0.0

    def to_onnx(self, output_path: str) -> str:
        """Export model to ONNX format for cross-platform deployment."""
        if not self._is_loaded:
            raise RuntimeError("Model must be loaded before ONNX export.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(
            1, 3, *self.config.input_size, device=self.device
        )

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"Model exported to ONNX: {output_path}")
        return str(output_path)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.config.model_name}, "
            f"device={self.device}, "
            f"loaded={self._is_loaded})"
        )
