"""
Unit Tests for Edge AI Model Classes
=====================================
Author: Pranay M Mahendrakar
Description: Comprehensive unit tests for base model, YOLO detector,
             MobileNet classifier, and EfficientNet classifier.

Run with:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v --tb=short
    pytest tests/test_models.py -v -k "test_yolo" --no-header
"""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from typing import List


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_frame():
    """Generate a dummy BGR frame for testing."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def small_frame():
    """Small test frame (224x224) for classification models."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def batch_frames():
    """Batch of test frames."""
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]


@pytest.fixture
def model_config_cpu():
    """CPU model configuration."""
    from models.base_model import ModelConfig
    return ModelConfig(device="cpu", confidence_threshold=0.5)


# ─── BaseModel Tests ─────────────────────────────────────────────────────────

class TestBaseModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_config(self):
        """Test default ModelConfig values."""
        from models.base_model import ModelConfig
        config = ModelConfig()
        assert config.device == "cpu"
        assert config.confidence_threshold == 0.5
        assert config.nms_threshold == 0.45
        assert config.max_detections == 100
        assert config.batch_size == 1
        assert config.num_classes == 80

    def test_custom_config(self):
        """Test custom ModelConfig initialization."""
        from models.base_model import ModelConfig
        config = ModelConfig(
            model_name="yolov8n",
            device="cuda",
            confidence_threshold=0.3,
            input_size=(416, 416),
        )
        assert config.model_name == "yolov8n"
        assert config.confidence_threshold == 0.3
        assert config.input_size == (416, 416)


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_default_result(self):
        """Test default InferenceResult values."""
        from models.base_model import InferenceResult
        result = InferenceResult()
        assert result.detections == []
        assert result.classifications == []
        assert result.inference_time_ms == 0.0
        assert result.frame_id == 0

    def test_total_latency(self):
        """Test total latency calculation."""
        from models.base_model import InferenceResult
        result = InferenceResult(
            preprocessing_time_ms=5.0,
            inference_time_ms=20.0,
            postprocessing_time_ms=3.0,
        )
        assert result.total_latency_ms == pytest.approx(28.0)

    def test_fps_calculation(self):
        """Test FPS calculation from latency."""
        from models.base_model import InferenceResult
        result = InferenceResult(inference_time_ms=33.33)
        # 33.33ms → ~30 FPS
        assert result.fps == pytest.approx(1000.0 / 33.33, rel=0.01)

    def test_fps_zero_latency(self):
        """Test FPS with zero latency doesn't raise."""
        from models.base_model import InferenceResult
        result = InferenceResult(inference_time_ms=0.0)
        assert result.fps == 0.0


# ─── YOLO Detector Tests ──────────────────────────────────────────────────────

class TestYOLODetector:
    """Tests for YOLODetector class."""

    def test_initialization_nano(self):
        """Test YOLODetector can be initialized with nano variant."""
        from models.yolo_detector import YOLODetector
        detector = YOLODetector(model_size="nano")
        assert detector.model_size == "nano"
        assert detector.yolo_model_name == "yolov8n"
        assert not detector.is_loaded

    def test_initialization_small(self):
        """Test YOLODetector small variant initialization."""
        from models.yolo_detector import YOLODetector
        detector = YOLODetector(model_size="small")
        assert detector.yolo_model_name == "yolov8s"

    def test_invalid_model_size(self):
        """Test that invalid model size raises ValueError."""
        from models.yolo_detector import YOLODetector
        with pytest.raises(ValueError, match="model_size must be one of"):
            YOLODetector(model_size="invalid")

    def test_predict_without_loading_raises(self, dummy_frame):
        """Test that calling predict without loading raises RuntimeError."""
        from models.yolo_detector import YOLODetector
        detector = YOLODetector(model_size="nano")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            detector.predict(dummy_frame)

    def test_repr(self):
        """Test string representation."""
        from models.yolo_detector import YOLODetector
        detector = YOLODetector(model_size="nano")
        repr_str = repr(detector)
        assert "YOLODetector" in repr_str
        assert "yolov8n" in repr_str

    def test_performance_stats_no_runs(self):
        """Test performance stats with no inference runs."""
        from models.yolo_detector import YOLODetector
        detector = YOLODetector(model_size="nano")
        stats = detector.get_performance_stats()
        assert stats["total_frames"] == 0
        assert stats["avg_latency_ms"] == 0.0

    def test_postprocess_empty_detections(self, dummy_frame):
        """Test postprocessing with empty detections returns valid result."""
        from models.yolo_detector import YOLODetector
        from models.base_model import InferenceResult
        detector = YOLODetector(model_size="nano")

        # Mock the raw output as empty
        mock_result = MagicMock()
        mock_result.boxes = None
        raw_output = [mock_result]

        result = detector.postprocess(raw_output, (480, 640))
        assert isinstance(result, InferenceResult)
        assert result.detections == []

    def test_draw_detections_shape(self, dummy_frame):
        """Test that draw_detections returns frame with same shape."""
        from models.yolo_detector import YOLODetector
        from models.base_model import InferenceResult
        detector = YOLODetector(model_size="nano")
        result = InferenceResult(
            detections=[{
                "class_id": 0, "class_name": "person",
                "confidence": 0.9,
                "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 300,
                         "width": 100, "height": 200,
                         "center_x": 150, "center_y": 200}
            }],
            inference_time_ms=25.0,
        )
        annotated = detector.draw_detections(dummy_frame, result)
        assert annotated.shape == dummy_frame.shape


# ─── MobileNet Tests ──────────────────────────────────────────────────────────

class TestMobileNetClassifier:
    """Tests for MobileNetClassifier class."""

    def test_initialization_small(self):
        """Test MobileNetClassifier small variant initialization."""
        from models.mobilenet_classifier import MobileNetClassifier
        classifier = MobileNetClassifier(variant="small")
        assert classifier.variant == "small"
        assert classifier.config.input_size == (224, 224)
        assert not classifier.is_loaded

    def test_initialization_large(self):
        """Test MobileNetClassifier large variant initialization."""
        from models.mobilenet_classifier import MobileNetClassifier
        classifier = MobileNetClassifier(variant="large")
        assert classifier.variant == "large"

    def test_invalid_variant(self):
        """Test invalid variant raises ValueError."""
        from models.mobilenet_classifier import MobileNetClassifier
        with pytest.raises(ValueError, match="variant must be one of"):
            MobileNetClassifier(variant="xlarge")

    def test_predict_without_loading(self, small_frame):
        """Test predict without loading raises RuntimeError."""
        from models.mobilenet_classifier import MobileNetClassifier
        classifier = MobileNetClassifier(variant="small")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            classifier.predict(small_frame)

    def test_top_k_configuration(self):
        """Test top_k parameter is stored correctly."""
        from models.mobilenet_classifier import MobileNetClassifier
        classifier = MobileNetClassifier(variant="small", top_k=3)
        assert classifier.top_k == 3

    def test_repr(self):
        """Test string representation."""
        from models.mobilenet_classifier import MobileNetClassifier
        classifier = MobileNetClassifier(variant="small")
        assert "MobileNetClassifier" in repr(classifier)


# ─── EfficientNet Tests ───────────────────────────────────────────────────────

class TestEfficientNetClassifier:
    """Tests for EfficientNetClassifier class."""

    def test_initialization_b0(self):
        """Test EfficientNetClassifier B0 variant initialization."""
        from models.efficientnet_classifier import EfficientNetClassifier
        classifier = EfficientNetClassifier(variant="b0")
        assert classifier.variant == "b0"
        assert classifier.config.input_size == (224, 224)

    def test_initialization_b1(self):
        """Test EfficientNetClassifier B1 variant initialization."""
        from models.efficientnet_classifier import EfficientNetClassifier
        classifier = EfficientNetClassifier(variant="b1")
        assert classifier.config.input_size == (240, 240)

    def test_invalid_variant(self):
        """Test invalid variant raises ValueError."""
        from models.efficientnet_classifier import EfficientNetClassifier
        with pytest.raises(ValueError, match="variant must be one of"):
            EfficientNetClassifier(variant="b99")

    def test_compound_scaling_b0(self):
        """Test compound scaling info for B0 (baseline)."""
        from models.efficientnet_classifier import EfficientNetClassifier
        classifier = EfficientNetClassifier(variant="b0")
        info = classifier.get_compound_scaling_info()
        assert info["phi"] == 0
        assert info["depth"] == 1.0
        assert info["width"] == 1.0

    def test_compound_scaling_b3(self):
        """Test compound scaling info for B3."""
        from models.efficientnet_classifier import EfficientNetClassifier
        classifier = EfficientNetClassifier(variant="b3")
        info = classifier.get_compound_scaling_info()
        assert info["phi"] == 2

    def test_predict_without_loading(self, small_frame):
        """Test predict without loading raises RuntimeError."""
        from models.efficientnet_classifier import EfficientNetClassifier
        classifier = EfficientNetClassifier(variant="b0")
        with pytest.raises(RuntimeError):
            classifier.predict(small_frame)

    def test_top_k_default(self):
        """Test default top_k value."""
        from models.efficientnet_classifier import EfficientNetClassifier
        classifier = EfficientNetClassifier(variant="b0")
        assert classifier.top_k == 5


# ─── Factory Function Tests ───────────────────────────────────────────────────

class TestFactoryFunctions:
    """Tests for model factory convenience functions."""

    def test_create_yolo_detector(self):
        """Test create_detector factory function."""
        from models.yolo_detector import create_detector
        detector = create_detector(model_size="nano", device="cpu", confidence=0.4)
        assert detector.model_size == "nano"
        assert detector.config.confidence_threshold == 0.4

    def test_create_mobilenet_classifier(self):
        """Test create_classifier factory function."""
        from models.mobilenet_classifier import create_classifier
        classifier = create_classifier(variant="small", device="cpu", top_k=3)
        assert classifier.variant == "small"
        assert classifier.top_k == 3

    def test_create_efficientnet_classifier(self):
        """Test create_efficientnet factory function."""
        from models.efficientnet_classifier import create_efficientnet
        classifier = create_efficientnet(variant="b0", device="cpu")
        assert classifier.variant == "b0"


# ─── Integration Tests (require model loading) ─────────────────────────────

@pytest.mark.slow
class TestModelIntegration:
    """Integration tests requiring actual model downloads. Marked as slow."""

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("ultralytics"),
        reason="ultralytics not installed"
    )
    def test_yolo_load_and_predict(self, dummy_frame):
        """Integration test: load YOLOv8n and run inference."""
        from models.yolo_detector import YOLODetector
        detector = YOLODetector(model_size="nano")
        detector.load_model()
        assert detector.is_loaded

        result = detector.predict(dummy_frame)
        assert result.total_latency_ms > 0
        assert result.fps > 0
        assert isinstance(result.detections, list)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torchvision"),
        reason="torchvision not installed"
    )
    def test_mobilenet_load_and_predict(self, small_frame):
        """Integration test: load MobileNetV3 and run inference."""
        from models.mobilenet_classifier import MobileNetClassifier
        classifier = MobileNetClassifier(variant="small")
        classifier.load_model(pretrained=True)
        assert classifier.is_loaded

        result = classifier.predict(small_frame)
        assert len(result.classifications) > 0
        assert result.classifications[0]["confidence"] > 0
        assert result.total_latency_ms > 0


# ─── Run Tests ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
