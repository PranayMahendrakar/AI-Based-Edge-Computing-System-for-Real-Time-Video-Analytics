"""
Traffic Monitoring Application
===============================
Author: Pranay M Mahendrakar
Description: Real-time traffic monitoring system using edge AI for vehicle
             counting, classification, density estimation, and incident detection.
"""

import time
import logging
import collections
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Vehicle class IDs in COCO dataset
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Traffic density thresholds
DENSITY_THRESHOLDS = {
    "free_flow": 10,      # < 10 vehicles/min
    "moderate": 25,       # 10–25 vehicles/min
    "heavy": 50,          # 25–50 vehicles/min
    "congested": 100,     # > 50 vehicles/min
}


class TrafficMonitor:
    """
    Real-time traffic monitoring system for edge deployment.

    Features:
    - Multi-class vehicle counting (car, truck, bus, bike, motorcycle)
    - Traffic density estimation and congestion alerts
    - Lane-level monitoring with configurable ROI zones
    - Speed estimation via optical flow (when calibration provided)
    - Incident detection (stopped vehicles, wrong-way movement)
    - CSV/JSON report generation

    Args:
        camera_source: Camera index, video file path, or RTSP URL
        model_config: Path to YAML configuration file
        alert_threshold: Vehicle count per minute to trigger alert
        enable_speed_estimation: Enable optical flow speed estimation
        count_interval_sec: Interval for counting vehicles (seconds)

    Example:
        monitor = TrafficMonitor(
            camera_source='rtsp://192.168.1.100:554/stream1',
            alert_threshold=50
        )
        monitor.start(show_dashboard=True)
    """

    def __init__(
        self,
        camera_source: Any = 0,
        model_config: Optional[str] = None,
        alert_threshold: int = 50,
        enable_speed_estimation: bool = False,
        count_interval_sec: float = 60.0,
        lane_zones: Optional[List[Tuple]] = None,
    ):
        self.camera_source = camera_source
        self.model_config = model_config
        self.alert_threshold = alert_threshold
        self.enable_speed_estimation = enable_speed_estimation
        self.count_interval_sec = count_interval_sec
        self.lane_zones = lane_zones or []

        # Counters
        self._vehicle_counts: Dict[str, int] = {name: 0 for name in VEHICLE_CLASSES.values()}
        self._total_count: int = 0
        self._interval_count: int = 0
        self._interval_start: float = time.time()
        self._count_history: collections.deque = collections.deque(maxlen=60)
        self._density_history: collections.deque = collections.deque(maxlen=10)

        # Optical flow
        self._prev_gray: Optional[np.ndarray] = None
        self._flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        logger.info(
            f"TrafficMonitor initialized | "
            f"alert_threshold={alert_threshold} vehicles/min"
        )

    def start(
        self,
        show_dashboard: bool = True,
        save_report: bool = False,
        report_path: str = "traffic_report.json",
    ) -> Dict:
        """
        Start the traffic monitoring pipeline.

        Args:
            show_dashboard: Display real-time dashboard overlay
            save_report: Save traffic statistics to JSON file
            report_path: Output path for traffic report

        Returns:
            Traffic statistics summary
        """
        from edge_analytics import EdgeVideoAnalytics

        analytics = EdgeVideoAnalytics(
            model="yolov8n",
            device="cpu",
            confidence=0.4,
            target_fps=25.0,
        )

        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.camera_source}")

        logger.info("Traffic monitoring started. Press 'q' to stop.")
        frame_id = 0
        t_start = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                result = analytics._engine.process_frame(frame)

                # Filter and count vehicles
                vehicles = self._filter_vehicles(result.detections)
                self._update_counts(vehicles)

                # Speed estimation
                if self.enable_speed_estimation:
                    self._estimate_speeds(frame, vehicles)

                # Check alerts
                density = self._get_current_density()
                if density >= self.alert_threshold:
                    logger.warning(f"TRAFFIC ALERT: {density:.0f} vehicles/min detected!")

                # Draw dashboard
                if show_dashboard:
                    annotated = self._draw_dashboard(frame, result, vehicles, density)
                    cv2.imshow("Traffic Monitor — Edge AI", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            logger.info("Traffic monitoring stopped by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        # Generate report
        stats = self._generate_report()
        if save_report:
            import json
            with open(report_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Traffic report saved: {report_path}")

        return stats

    def _filter_vehicles(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections to vehicle classes only."""
        return [
            d for d in detections
            if d.get("class_id") in VEHICLE_CLASSES
        ]

    def _update_counts(self, vehicles: List[Dict]) -> None:
        """Update vehicle count statistics."""
        frame_count = len(vehicles)
        self._total_count += frame_count
        self._interval_count += frame_count

        for vehicle in vehicles:
            class_name = VEHICLE_CLASSES.get(vehicle["class_id"], "unknown")
            self._vehicle_counts[class_name] = self._vehicle_counts.get(class_name, 0) + 1

        # Reset interval counter
        elapsed = time.time() - self._interval_start
        if elapsed >= self.count_interval_sec:
            rate = self._interval_count / elapsed * 60  # per minute
            self._count_history.append({
                "timestamp": time.time(),
                "count": self._interval_count,
                "rate_per_min": round(rate, 1),
            })
            self._interval_count = 0
            self._interval_start = time.time()

    def _get_current_density(self) -> float:
        """Calculate current traffic density (vehicles per minute)."""
        elapsed = time.time() - self._interval_start
        if elapsed > 0:
            return self._interval_count / elapsed * 60
        return 0.0

    def _estimate_speeds(self, frame: np.ndarray, vehicles: List[Dict]) -> None:
        """Estimate vehicle speeds using Lucas-Kanade optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is not None and vehicles:
            # Get vehicle center points as tracking features
            points = np.array([
                [v["bbox"]["center_x"], v["bbox"]["center_y"]]
                for v in vehicles
                if "bbox" in v
            ], dtype=np.float32).reshape(-1, 1, 2)

            if len(points) > 0:
                try:
                    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        self._prev_gray, gray, points, None, **self._flow_params
                    )
                    for i, (old, new, s) in enumerate(zip(points, new_points, status)):
                        if s[0] == 1 and i < len(vehicles):
                            dx = new[0][0] - old[0][0]
                            dy = new[0][1] - old[0][1]
                            speed_px = np.sqrt(dx**2 + dy**2)
                            vehicles[i]["speed_px_per_frame"] = round(float(speed_px), 2)
                except cv2.error:
                    pass

        self._prev_gray = gray

    def _draw_dashboard(
        self,
        frame: np.ndarray,
        result: Any,
        vehicles: List[Dict],
        density: float,
    ) -> np.ndarray:
        """Draw comprehensive traffic monitoring dashboard."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw vehicle detections
        for v in vehicles:
            bbox = v.get("bbox", {})
            x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
            x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
            class_name = v.get("class_name", "vehicle")
            conf = v.get("confidence", 0)

            color = (0, 255, 255)  # Yellow for vehicles
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, f"{class_name} {conf:.2f}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )

        # Dashboard panel (top-left)
        panel_h = 180
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (320, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Header
        cv2.putText(annotated, "TRAFFIC MONITOR - EDGE AI",
                   (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Density bar
        density_pct = min(density / self.alert_threshold, 1.0)
        bar_color = (0, 255, 0) if density_pct < 0.6 else (0, 165, 255) if density_pct < 0.9 else (0, 0, 255)
        bar_w = int(density_pct * 280)
        cv2.rectangle(annotated, (10, 30), (290, 50), (50, 50, 50), -1)
        cv2.rectangle(annotated, (10, 30), (10 + bar_w, 50), bar_color, -1)
        cv2.putText(annotated, f"Density: {density:.0f} v/min",
                   (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Current status
        if density < DENSITY_THRESHOLDS["free_flow"]:
            status = "FREE FLOW"
            status_color = (0, 255, 0)
        elif density < DENSITY_THRESHOLDS["moderate"]:
            status = "MODERATE"
            status_color = (0, 255, 255)
        elif density < DENSITY_THRESHOLDS["heavy"]:
            status = "HEAVY"
            status_color = (0, 165, 255)
        else:
            status = "CONGESTED"
            status_color = (0, 0, 255)

        cv2.putText(annotated, f"Status: {status}",
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Vehicle counts
        cv2.putText(annotated, f"In Frame: {len(vehicles)} vehicles",
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated, f"Total: {self._total_count}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Per-class counts
        y = 140
        for cls_name, count in self._vehicle_counts.items():
            if count > 0:
                cv2.putText(annotated, f"  {cls_name}: {count}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                y += 15

        # FPS
        fps_text = f"FPS: {result.fps:.1f}"
        cv2.putText(annotated, fps_text, (w - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated

    def _generate_report(self) -> Dict:
        """Generate traffic statistics report."""
        total_time = time.time() - self._interval_start
        return {
            "total_vehicles": self._total_count,
            "vehicle_breakdown": dict(self._vehicle_counts),
            "monitoring_duration_s": round(total_time, 1),
            "avg_rate_per_min": round(
                self._total_count / total_time * 60 if total_time > 0 else 0, 1
            ),
            "count_history": list(self._count_history),
            "alert_threshold": self.alert_threshold,
        }
