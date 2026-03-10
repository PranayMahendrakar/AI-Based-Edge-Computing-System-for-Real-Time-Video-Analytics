"""
EdgeVideoAnalytics — Main System Entry Point
============================================
Author: Pranay M Mahendrakar
Description: High-level API for the AI-Based Edge Computing System for
             Real-Time Video Analytics. Provides a unified interface for
             running YOLO, MobileNet, and EfficientNet models on edge devices.
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EdgeVideoAnalytics:
    """
    High-level API for Edge Computing Video Analytics System.

    This class provides a simple, unified interface to run the complete
    AI-based video analytics pipeline on edge devices, supporting:
    - Real-time object detection (YOLOv8)
    - Image classification (MobileNetV3, EfficientNet)
    - Smart surveillance, traffic monitoring, robot perception

    Args:
        model: Model type ('yolov8n' | 'yolov8s' | 'mobilenet' | 'efficientnet')
        device: Compute device ('cpu' | 'cuda' | 'trt')
        confidence: Detection confidence threshold (0.0–1.0)
        target_fps: Target frames per second
        applications: List of applications to enable
        verbose: Enable detailed logging

    Example:
        analytics = EdgeVideoAnalytics(
            model='yolov8n',
            device='cuda',
            confidence=0.5,
            target_fps=30
        )
        analytics.process_stream(source=0, show=True)
    """

    SUPPORTED_MODELS = {
        "yolov8n": ("yolo", "nano"),
        "yolov8s": ("yolo", "small"),
        "mobilenet": ("mobilenet", "small"),
        "mobilenet_large": ("mobilenet", "large"),
        "efficientnet": ("efficientnet", "b0"),
        "efficientnet_b1": ("efficientnet", "b1"),
    }

    def __init__(
        self,
        model: str = "yolov8n",
        device: str = "cpu",
        confidence: float = 0.5,
        target_fps: float = 30.0,
        applications: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model
        self.device = device
        self.confidence = confidence
        self.target_fps = target_fps
        self.applications = applications or []
        self.verbose = verbose

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Build pipeline
        self._engine = self._build_pipeline()
        logger.info(
            f"EdgeVideoAnalytics ready | "
            f"model={model} | device={device} | fps={target_fps}"
        )

    def _build_pipeline(self):
        """Construct the inference engine pipeline based on model selection."""
        from inference.edge_inference_engine import EdgeInferenceEngine
        from models.yolo_detector import YOLODetector
        from models.mobilenet_classifier import MobileNetClassifier
        from models.efficientnet_classifier import EfficientNetClassifier
        from models.base_model import ModelConfig

        engine = EdgeInferenceEngine(
            device=self.device,
            target_fps=self.target_fps,
            verbose=self.verbose,
        )

        model_type, variant = self.SUPPORTED_MODELS[self.model_name]
        config = ModelConfig(
            device=self.device,
            confidence_threshold=self.confidence,
            half_precision=(self.device == "cuda"),
        )

        if model_type == "yolo":
            detector = YOLODetector(model_size=variant, config=config)
            engine.add_stage("detect", detector, timeout_ms=80)

        elif model_type == "mobilenet":
            classifier = MobileNetClassifier(variant=variant, config=config)
            engine.add_stage("classify", classifier, timeout_ms=50)

        elif model_type == "efficientnet":
            classifier = EfficientNetClassifier(variant=variant, config=config)
            engine.add_stage("classify", classifier, timeout_ms=60)

        # Load and warmup
        engine.load_all()
        engine.warmup_all()

        return engine

    def process_stream(
        self,
        source: Union[int, str] = 0,
        show: bool = True,
        save_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """
        Process a live camera stream or video file.

        Args:
            source: Camera index (0, 1, ...) or video path or RTSP URL
            show: Display real-time annotated video
            save_path: Save annotated output to this path
            max_frames: Limit number of frames (None = unlimited)

        Returns:
            Dictionary with performance statistics
        """
        logger.info(f"Processing stream: {source}")
        results = self._engine.run(
            video_path=source,
            show=show,
            save_output=save_path,
            max_frames=max_frames,
        )
        self._log_summary(results)
        return results

    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image file.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with detection/classification results
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")

        result = self._engine.process_frame(frame)

        output = {
            "image": image_path,
            "detections": result.detections,
            "classifications": result.classifications,
            "latency_ms": result.total_latency_ms,
            "fps": result.fps,
        }

        logger.info(
            f"Image processed: {len(result.detections)} detections | "
            f"{result.total_latency_ms:.1f}ms"
        )
        return output

    def get_stats(self) -> Dict:
        """Return current performance statistics."""
        return self._engine.get_statistics(include_stages=True)

    def _log_summary(self, stats: Dict) -> None:
        """Log performance summary."""
        pipeline = stats.get("pipeline", {})
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total Frames:    {pipeline.get('total_frames', 0)}")
        logger.info(f"  Dropped Frames:  {pipeline.get('dropped_frames', 0)}")
        logger.info(f"  Average FPS:     {pipeline.get('avg_fps', 0):.1f}")
        logger.info(f"  Avg Latency:     {pipeline.get('avg_latency_ms', 0):.1f}ms")
        logger.info(f"  P95 Latency:     {pipeline.get('p95_latency_ms', 0):.1f}ms")
        logger.info("=" * 60)


def main():
    """Command-line interface for EdgeVideoAnalytics."""
    parser = argparse.ArgumentParser(
        description="AI-Based Edge Computing System for Real-Time Video Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam detection with YOLOv8 nano
  python edge_analytics.py --model yolov8n --source 0 --device cuda

  # Video file with MobileNet
  python edge_analytics.py --model mobilenet --source video.mp4 --save output.mp4

  # RTSP stream
  python edge_analytics.py --model yolov8s --source rtsp://192.168.1.100:554/stream

  # Single image
  python edge_analytics.py --model yolov8n --image photo.jpg
        """
    )

    parser.add_argument(
        "--model", type=str, default="yolov8n",
        choices=list(EdgeVideoAnalytics.SUPPORTED_MODELS.keys()),
        help="Model to use for inference"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: camera index, file path, or RTSP URL"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Single image file to process"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "trt"],
        help="Compute device"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Target FPS"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save annotated output to this file"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Disable real-time display"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to process"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Initialize system
    analytics = EdgeVideoAnalytics(
        model=args.model,
        device=args.device,
        confidence=args.confidence,
        target_fps=args.fps,
        verbose=args.verbose,
    )

    if args.image:
        # Process single image
        result = analytics.process_image(args.image)
        print(f"\nDetections: {len(result['detections'])}")
        for det in result["detections"]:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")
        if result["classifications"]:
            print(f"Top class: {result['classifications'][0]['class_name']}")
    else:
        # Process video stream
        source = args.source
        try:
            source = int(source)  # Camera index
        except ValueError:
            pass  # Keep as string (file path or URL)

        analytics.process_stream(
            source=source,
            show=not args.no_show,
            save_path=args.save,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
