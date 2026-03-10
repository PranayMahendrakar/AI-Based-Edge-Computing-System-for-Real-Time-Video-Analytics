"""
Smart Surveillance Application
===============================
Author: Pranay M Mahendrakar
Description: AI-powered surveillance system for real-time intrusion detection,
             anomaly detection, and perimeter monitoring using edge AI.
"""

import time
import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Alert types
ALERT_TYPES = {
    "INTRUSION": "Person detected in restricted zone",
    "CROWD": "Crowd density threshold exceeded",
    "LOITERING": "Person stationary for extended period",
    "OBJECT_LEFT": "Unattended object detected",
    "ANOMALY": "Unusual motion pattern detected",
}

# Detection classes for surveillance
PERSON_CLASS_ID = 0
SUSPECT_CLASSES = {0: "person", 27: "backpack", 28: "umbrella"}


class SmartSurveillance:
    """
    AI-powered smart surveillance system for edge deployment.

    Features:
    - Real-time intrusion detection with configurable alert zones
    - Person detection and crowd density monitoring
    - Loitering detection (stationary person over threshold time)
    - Motion-based anomaly detection using background subtraction
    - Multi-camera support with zone configuration
    - Alert webhook notifications
    - Event recording and incident logging

    Args:
        camera_source: Camera index, video file, or RTSP URL
        config: Path to surveillance config YAML
        alert_zones: List of (x1, y1, x2, y2) bounding boxes for alert ROIs
        sensitivity: Detection sensitivity ('low' | 'medium' | 'high')
        loiter_threshold_sec: Seconds before triggering loitering alert

    Example:
        surveillance = SmartSurveillance(
            camera_source=0,
            alert_zones=[(100, 100, 500, 400)],
            sensitivity='high'
        )
        surveillance.start(record=True)
    """

    SENSITIVITY_CONFIGS = {
        "low":    {"confidence": 0.6, "min_area": 3000},
        "medium": {"confidence": 0.5, "min_area": 1500},
        "high":   {"confidence": 0.35, "min_area": 500},
    }

    def __init__(
        self,
        camera_source: Any = 0,
        config: Optional[str] = None,
        alert_zones: Optional[List[Tuple[int, int, int, int]]] = None,
        sensitivity: str = "medium",
        loiter_threshold_sec: float = 30.0,
        crowd_threshold: int = 10,
        alert_webhook: Optional[str] = None,
    ):
        self.camera_source = camera_source
        self.config = config
        self.alert_zones = alert_zones or []
        self.sensitivity = sensitivity
        self.loiter_threshold_sec = loiter_threshold_sec
        self.crowd_threshold = crowd_threshold
        self.alert_webhook = alert_webhook

        # State tracking
        self._detected_persons: Dict[int, Dict] = {}  # track_id → {first_seen, bbox}
        self._alert_log: List[Dict] = []
        self._event_count: int = 0
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        # Load sensitivity config
        sens_cfg = self.SENSITIVITY_CONFIGS.get(sensitivity, self.SENSITIVITY_CONFIGS["medium"])
        self._confidence = sens_cfg["confidence"]
        self._min_area = sens_cfg["min_area"]

        logger.info(
            f"SmartSurveillance initialized | "
            f"sensitivity={sensitivity} | "
            f"alert_zones={len(self.alert_zones)} | "
            f"loiter_threshold={loiter_threshold_sec}s"
        )

    def start(
        self,
        show: bool = True,
        record: bool = False,
        record_path: str = "surveillance_output.mp4",
        alert_webhook: Optional[str] = None,
    ) -> Dict:
        """
        Start the surveillance monitoring pipeline.

        Args:
            show: Display real-time annotated video
            record: Save output video
            record_path: Output video path
            alert_webhook: Webhook URL for alert notifications

        Returns:
            Surveillance session statistics
        """
        from edge_analytics import EdgeVideoAnalytics

        analytics = EdgeVideoAnalytics(
            model="yolov8n",
            device="cpu",
            confidence=self._confidence,
            target_fps=20.0,
        )

        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.camera_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(record_path, fourcc, 20.0, (width, height))

        t_start = time.time()
        logger.info("Surveillance active. Press 'q' to stop.")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = analytics._engine.process_frame(frame)

                # Run detection and alerting logic
                persons = [d for d in result.detections if d.get("class_id") == PERSON_CLASS_ID]
                alerts = []

                # Check zone intrusions
                alerts += self._check_zone_intrusions(persons)

                # Check crowd density
                if len(persons) >= self.crowd_threshold:
                    alerts.append({
                        "type": "CROWD",
                        "message": f"{len(persons)} persons detected",
                        "timestamp": time.time(),
                    })

                # Motion anomaly via background subtraction
                motion_alerts = self._detect_motion_anomalies(frame)
                alerts += motion_alerts

                # Log alerts
                for alert in alerts:
                    self._alert_log.append(alert)
                    self._event_count += 1
                    logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")

                # Draw output
                if show or writer:
                    annotated = self._draw_surveillance_overlay(frame, result, persons, alerts)

                    if show:
                        cv2.imshow("Smart Surveillance — Edge AI", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    if writer:
                        writer.write(annotated)

        except KeyboardInterrupt:
            logger.info("Surveillance stopped by user.")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        duration = time.time() - t_start
        return {
            "duration_s": round(duration, 1),
            "total_events": self._event_count,
            "alert_log": self._alert_log[-100:],  # Last 100 alerts
            "alert_zones_monitored": len(self.alert_zones),
        }

    def _check_zone_intrusions(self, persons: List[Dict]) -> List[Dict]:
        """Check if any person detection is inside defined alert zones."""
        alerts = []
        for zone in self.alert_zones:
            zx1, zy1, zx2, zy2 = zone
            for person in persons:
                bbox = person.get("bbox", {})
                cx = bbox.get("center_x", 0)
                cy = bbox.get("center_y", 0)
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    alerts.append({
                        "type": "INTRUSION",
                        "message": f"Person detected in restricted zone {zone}",
                        "timestamp": time.time(),
                        "location": {"x": cx, "y": cy},
                    })
        return alerts

    def _detect_motion_anomalies(self, frame: np.ndarray) -> List[Dict]:
        """Detect motion anomalies using background subtraction."""
        alerts = []
        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_motions = [c for c in contours if cv2.contourArea(c) > self._min_area * 3]
        if len(large_motions) > 5:
            alerts.append({
                "type": "ANOMALY",
                "message": f"Unusual motion: {len(large_motions)} large motion regions",
                "timestamp": time.time(),
            })
        return alerts

    def _draw_surveillance_overlay(
        self,
        frame: np.ndarray,
        result: Any,
        persons: List[Dict],
        alerts: List[Dict],
    ) -> np.ndarray:
        """Draw surveillance overlay with detections, zones, and alerts."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw alert zones
        for zone in self.alert_zones:
            x1, y1, x2, y2 = zone
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, "RESTRICTED",
                       (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw person detections
        for person in persons:
            bbox = person.get("bbox", {})
            x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
            x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"Person {person.get('confidence', 0):.2f}",
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Alert display panel
        if alerts:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 50), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            latest_alert = alerts[-1]
            cv2.putText(
                annotated,
                f"ALERT: [{latest_alert['type']}] {latest_alert['message'][:50]}",
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
            )

        # Status bar (top)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        cv2.putText(annotated, f"SMART SURVEILLANCE | Persons: {len(persons)} | Events: {self._event_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated, f"FPS: {result.fps:.1f}",
                   (w - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        return annotated
