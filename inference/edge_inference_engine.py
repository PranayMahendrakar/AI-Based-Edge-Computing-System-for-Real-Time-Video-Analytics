"""
Edge Inference Engine — Multi-Stage Video Analytics Pipeline
============================================================
Author: Pranay M Mahendrakar
Description: Orchestrates multi-stage deep learning inference pipeline for
             real-time video analytics on edge devices. Manages model lifecycle,
             adaptive model selection, latency monitoring, and frame queuing.
"""

import time
import queue
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from models.base_model import BaseModel, InferenceResult, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Represents a single stage in the inference pipeline."""
    name: str
    model: BaseModel
    enabled: bool = True
    required: bool = True     # If False, pipeline continues on stage failure
    timeout_ms: float = 100.0 # Max time budget for this stage

    def __repr__(self) -> str:
        return f"PipelineStage(name={self.name}, model={self.model.model_name}, enabled={self.enabled})"


@dataclass
class PipelineResult:
    """Aggregated result from the complete inference pipeline."""
    frame_id: int = 0
    stage_results: Dict[str, InferenceResult] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    fps: float = 0.0
    timestamp: float = field(default_factory=time.time)
    dropped: bool = False
    error: Optional[str] = None

    @property
    def detections(self) -> List[Dict]:
        """Get detections from the 'detect' stage if available."""
        detect_result = self.stage_results.get("detect")
        return detect_result.detections if detect_result else []

    @property
    def classifications(self) -> List[Dict]:
        """Get classifications from the 'classify' or 'verify' stage."""
        for stage_name in ("verify", "classify"):
            result = self.stage_results.get(stage_name)
            if result and result.classifications:
                return result.classifications
        return []

    def get_stage(self, stage_name: str) -> Optional[InferenceResult]:
        return self.stage_results.get(stage_name)


class LatencyMonitor:
    """Tracks inference latency with rolling window statistics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._latencies: deque = deque(maxlen=window_size)
        self._fps_samples: deque = deque(maxlen=window_size)
        self._last_frame_time: float = 0.0

    def record(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)
        now = time.perf_counter()
        if self._last_frame_time > 0:
            frame_time = now - self._last_frame_time
            self._fps_samples.append(1.0 / frame_time if frame_time > 0 else 0)
        self._last_frame_time = now

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self._latencies)) if self._latencies else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self._latencies, 95)) if self._latencies else 0.0

    @property
    def avg_fps(self) -> float:
        return float(np.mean(self._fps_samples)) if self._fps_samples else 0.0

    @property
    def min_latency_ms(self) -> float:
        return float(np.min(self._latencies)) if self._latencies else 0.0

    @property
    def max_latency_ms(self) -> float:
        return float(np.max(self._latencies)) if self._latencies else 0.0

    def get_stats(self) -> Dict[str, float]:
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "avg_fps": round(self.avg_fps, 1),
            "sample_count": len(self._latencies),
        }


class EdgeInferenceEngine:
    """
    Multi-stage inference pipeline orchestrator for edge video analytics.

    The engine chains multiple models in sequence (detect → classify → verify)
    with latency monitoring, adaptive frame dropping, and performance logging.

    Key Features:
    - Multi-stage pipeline with named stages
    - Adaptive frame dropping when behind real-time
    - Per-stage and end-to-end latency tracking
    - Thread-safe frame queue with configurable buffer
    - Callback hooks for result processing
    - Graceful degradation on stage failures

    Args:
        device: Target device ('cpu' | 'cuda' | 'trt')
        target_fps: Target FPS for adaptive frame control
        max_queue_size: Maximum frame buffer size
        drop_frames: Enable adaptive frame dropping
        verbose: Enable detailed per-stage logging

    Example:
        engine = EdgeInferenceEngine(device='cuda', target_fps=30)
        engine.add_stage('detect',   YOLODetector(model_size='nano'))
        engine.add_stage('classify', MobileNetClassifier(variant='small'))

        # Load all models
        engine.load_all()
        engine.warmup_all()

        # Process video
        results = engine.run(video_path='traffic.mp4', show=True)
        stats = engine.get_statistics()
        print(f"Avg FPS: {stats['avg_fps']}, P95 Latency: {stats['p95_latency_ms']}ms")
    """

    def __init__(
        self,
        device: str = "cpu",
        target_fps: float = 30.0,
        max_queue_size: int = 10,
        drop_frames: bool = True,
        verbose: bool = False,
        result_callback: Optional[Callable[[PipelineResult], None]] = None,
    ):
        self.device = device
        self.target_fps = target_fps
        self.max_queue_size = max_queue_size
        self.drop_frames = drop_frames
        self.verbose = verbose
        self.result_callback = result_callback

        self._stages: List[PipelineStage] = []
        self._frame_id: int = 0
        self._latency_monitor = LatencyMonitor(window_size=100)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._is_running: bool = False
        self._total_frames: int = 0
        self._dropped_frames: int = 0

        self._target_frame_time = 1.0 / target_fps

        logger.info(
            f"EdgeInferenceEngine initialized | "
            f"device={device} | target_fps={target_fps} | "
            f"drop_frames={drop_frames}"
        )

    def add_stage(
        self,
        name: str,
        model: BaseModel,
        enabled: bool = True,
        required: bool = True,
        timeout_ms: float = 100.0,
    ) -> "EdgeInferenceEngine":
        """
        Add a processing stage to the pipeline.

        Args:
            name: Unique stage identifier (e.g., 'detect', 'classify', 'verify')
            model: Initialized model instance (not yet loaded)
            enabled: Whether this stage is active
            required: If False, errors in this stage don't abort the pipeline
            timeout_ms: Maximum time budget for this stage

        Returns:
            self (for method chaining)
        """
        # Check for duplicate names
        if any(s.name == name for s in self._stages):
            raise ValueError(f"Stage '{name}' already exists in pipeline.")

        stage = PipelineStage(
            name=name,
            model=model,
            enabled=enabled,
            required=required,
            timeout_ms=timeout_ms,
        )
        self._stages.append(stage)
        logger.info(f"Stage added: [{name}] → {model.model_name}")
        return self

    def load_all(self, model_paths: Optional[Dict[str, str]] = None) -> None:
        """
        Load all stage models.

        Args:
            model_paths: Optional dict mapping stage names to model weight paths
        """
        logger.info(f"Loading {len(self._stages)} pipeline stages...")
        paths = model_paths or {}

        for stage in self._stages:
            if not stage.enabled:
                logger.info(f"  Skipping disabled stage: {stage.name}")
                continue
            try:
                t = time.perf_counter()
                stage.model.load_model(paths.get(stage.name))
                elapsed = (time.perf_counter() - t) * 1000
                logger.info(f"  [{stage.name}] loaded in {elapsed:.0f}ms")
            except Exception as e:
                logger.error(f"  [{stage.name}] failed to load: {e}")
                if stage.required:
                    raise

    def warmup_all(self) -> None:
        """Run warmup on all loaded pipeline stages."""
        logger.info("Running pipeline warmup...")
        for stage in self._stages:
            if stage.enabled and stage.model.is_loaded:
                logger.info(f"  Warming up: {stage.name}")
                stage.model.warmup()
        logger.info("Warmup complete for all stages.")

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """
        Run the complete pipeline on a single frame.

        Args:
            frame: Input BGR frame as numpy array

        Returns:
            PipelineResult with all stage outputs and latency metrics
        """
        t_pipeline_start = time.perf_counter()
        self._frame_id += 1
        pipeline_result = PipelineResult(frame_id=self._frame_id)

        for stage in self._stages:
            if not stage.enabled:
                continue

            if not stage.model.is_loaded:
                if stage.required:
                    raise RuntimeError(f"Stage '{stage.name}' model not loaded.")
                logger.warning(f"Stage '{stage.name}' skipped (not loaded)")
                continue

            try:
                t_stage = time.perf_counter()
                stage_result = stage.model.predict(frame, frame_id=self._frame_id)
                stage_elapsed = (time.perf_counter() - t_stage) * 1000

                pipeline_result.stage_results[stage.name] = stage_result

                if self.verbose:
                    n_det = len(stage_result.detections)
                    n_cls = len(stage_result.classifications)
                    logger.debug(
                        f"  [{stage.name}] {stage_elapsed:.1f}ms | "
                        f"detections={n_det} | classifications={n_cls}"
                    )

                # Check timeout
                if stage_elapsed > stage.timeout_ms:
                    logger.warning(
                        f"Stage '{stage.name}' exceeded timeout: "
                        f"{stage_elapsed:.1f}ms > {stage.timeout_ms}ms"
                    )

            except Exception as e:
                logger.error(f"Stage '{stage.name}' error: {e}")
                pipeline_result.error = str(e)
                if stage.required:
                    raise
                # Continue pipeline for non-required stages

        # Calculate total latency
        total_elapsed = (time.perf_counter() - t_pipeline_start) * 1000
        pipeline_result.total_latency_ms = total_elapsed
        pipeline_result.fps = 1000.0 / total_elapsed if total_elapsed > 0 else 0.0

        # Record latency
        self._latency_monitor.record(total_elapsed)
        self._total_frames += 1

        # Invoke callback if registered
        if self.result_callback is not None:
            try:
                self.result_callback(pipeline_result)
            except Exception as e:
                logger.warning(f"Result callback error: {e}")

        return pipeline_result

    def run(
        self,
        video_path: Union[str, int],
        show: bool = False,
        save_output: Optional[str] = None,
        max_frames: Optional[int] = None,
        draw_fn: Optional[Callable[[np.ndarray, PipelineResult], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Run the inference pipeline on a video source.

        Args:
            video_path: Path to video file, or integer camera index, or RTSP URL
            show: Display results in real-time using cv2.imshow
            save_output: Path to save annotated output video (e.g., 'output.mp4')
            max_frames: Maximum number of frames to process (None = all)
            draw_fn: Optional function(frame, result) → annotated_frame

        Returns:
            Dictionary with performance statistics and summary metrics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_path}")

        # Get video properties
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video source: {video_path} | "
            f"{width}x{height} @ {fps_src:.1f}fps | "
            f"frames={total_frames_src}"
        )

        # Setup video writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_output, fourcc, min(self.target_fps, fps_src), (width, height))
            logger.info(f"Saving output to: {save_output}")

        frame_count = 0
        t_start = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_count >= max_frames:
                    break

                # Adaptive frame dropping
                if self.drop_frames:
                    elapsed = time.perf_counter() - t_start
                    expected_frames = elapsed / self._target_frame_time
                    if frame_count < expected_frames - 2:
                        self._dropped_frames += 1
                        frame_count += 1
                        continue

                # Process frame
                result = self.process_frame(frame)
                frame_count += 1

                # Annotate frame
                if show or writer:
                    annotated = frame.copy()
                    if draw_fn:
                        annotated = draw_fn(annotated, result)
                    else:
                        annotated = self._default_draw(annotated, result)

                    if show:
                        cv2.imshow("Edge Analytics Pipeline", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.info("User requested stop (q key).")
                            break

                    if writer:
                        writer.write(annotated)

                # Live stats every 50 frames
                if frame_count % 50 == 0:
                    stats = self._latency_monitor.get_stats()
                    logger.info(
                        f"Frame {frame_count}/{total_frames_src} | "
                        f"FPS={stats['avg_fps']:.1f} | "
                        f"Latency={stats['avg_latency_ms']:.1f}ms"
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        total_time = time.perf_counter() - t_start
        logger.info(
            f"Processing complete: {frame_count} frames in {total_time:.1f}s | "
            f"Effective FPS: {frame_count/total_time:.1f}"
        )

        return self.get_statistics(include_stages=True)

    def _default_draw(self, frame: np.ndarray, result: PipelineResult) -> np.ndarray:
        """Default annotation: draw bounding boxes and FPS overlay."""
        annotated = frame.copy()

        # Draw detections
        for det in result.detections:
            bbox = det.get("bbox", {})
            x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
            x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Top classification
        if result.classifications:
            top = result.classifications[0]
            cls_text = f"Class: {top['class_name']} ({top['confidence_pct']:.1f}%)"
            cv2.putText(annotated, cls_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # FPS overlay
        fps_text = f"FPS: {result.fps:.1f} | Lat: {result.total_latency_ms:.1f}ms"
        cv2.putText(annotated, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated

    def get_statistics(self, include_stages: bool = False) -> Dict[str, Any]:
        """Return comprehensive performance statistics."""
        stats = {
            "pipeline": {
                **self._latency_monitor.get_stats(),
                "total_frames": self._total_frames,
                "dropped_frames": self._dropped_frames,
                "drop_rate_pct": round(
                    100 * self._dropped_frames / max(self._total_frames, 1), 1
                ),
                "stages": len(self._stages),
                "device": self.device,
            }
        }

        if include_stages:
            stage_stats = {}
            for stage in self._stages:
                if stage.model.is_loaded:
                    stage_stats[stage.name] = stage.model.get_performance_stats()
            stats["stages"] = stage_stats

        return stats

    def enable_stage(self, name: str) -> None:
        for stage in self._stages:
            if stage.name == name:
                stage.enabled = True
                logger.info(f"Stage enabled: {name}")
                return
        raise ValueError(f"Stage '{name}' not found.")

    def disable_stage(self, name: str) -> None:
        for stage in self._stages:
            if stage.name == name:
                stage.enabled = False
                logger.info(f"Stage disabled: {name}")
                return
        raise ValueError(f"Stage '{name}' not found.")

    def __repr__(self) -> str:
        stage_names = " → ".join(s.name for s in self._stages if s.enabled)
        return (
            f"EdgeInferenceEngine("
            f"stages=[{stage_names}], "
            f"device={self.device}, "
            f"target_fps={self.target_fps})"
        )
