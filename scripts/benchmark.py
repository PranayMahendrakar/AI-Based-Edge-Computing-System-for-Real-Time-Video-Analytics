"""
Performance Benchmarking Tool for Edge AI Video Analytics
==========================================================
Author: Pranay M Mahendrakar
Description: Comprehensive benchmarking utility for measuring inference latency,
             throughput, memory usage, and accuracy metrics across all supported
             models on edge devices.
"""

import time
import argparse
import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def get_device_info() -> Dict:
    """Collect device hardware information."""
    import platform
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or "Unknown",
    }

    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        info["pytorch_version"] = "Not installed"

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["cpu_cores"] = psutil.cpu_count()
        info["cpu_freq_mhz"] = round(psutil.cpu_freq().max if psutil.cpu_freq() else 0, 0)
    except ImportError:
        pass

    return info


def generate_test_frames(count: int = 100, resolution: Tuple[int, int] = (640, 640)) -> List[np.ndarray]:
    """Generate synthetic test frames for benchmarking."""
    frames = []
    h, w = resolution
    for i in range(count):
        # Generate varied synthetic frames (not all zeros)
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        # Add some structure to make it more realistic
        cv2.circle(frame, (w//2, h//2), min(w, h)//4, (255, 128, 0), -1)
        frames.append(frame)
    return frames


def benchmark_model(
    model_name: str,
    device: str,
    num_warmup: int = 10,
    num_iterations: int = 100,
    resolution: Tuple[int, int] = (640, 640),
) -> Dict:
    """
    Benchmark a single model for latency and throughput.

    Args:
        model_name: Model identifier
        device: 'cpu' | 'cuda'
        num_warmup: Warmup iterations (excluded from stats)
        num_iterations: Benchmark iterations
        resolution: Input frame resolution (H, W)

    Returns:
        Benchmark results dictionary
    """
    logger.info(f"Benchmarking: {model_name} | device={device} | iterations={num_iterations}")

    # Import model
    try:
        if "yolo" in model_name:
            from models.yolo_detector import YOLODetector
            from models.base_model import ModelConfig
            size_map = {"yolov8n": "nano", "yolov8s": "small"}
            model = YOLODetector(
                model_size=size_map.get(model_name, "nano"),
                config=ModelConfig(device=device, half_precision=(device == "cuda")),
            )
        elif "mobilenet" in model_name:
            from models.mobilenet_classifier import MobileNetClassifier
            from models.base_model import ModelConfig
            variant = "large" if "large" in model_name else "small"
            model = MobileNetClassifier(
                variant=variant,
                config=ModelConfig(device=device, half_precision=(device == "cuda")),
            )
        elif "efficientnet" in model_name:
            from models.efficientnet_classifier import EfficientNetClassifier
            from models.base_model import ModelConfig
            variant = model_name.replace("efficientnet_", "").replace("efficientnet-", "")
            variant = variant if variant in ["b0", "b1", "b2", "b3"] else "b0"
            model = EfficientNetClassifier(
                variant=variant,
                config=ModelConfig(device=device, half_precision=(device == "cuda")),
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.load_model()
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return {"model": model_name, "status": "FAILED", "error": str(e)}

    # Generate test frames
    frames = generate_test_frames(max(num_warmup, num_iterations), resolution)

    # Warmup
    logger.info(f"  Warming up ({num_warmup} iterations)...")
    for i in range(num_warmup):
        model.predict(frames[i % len(frames)])
    model.reset_stats()

    # Benchmark
    logger.info(f"  Running benchmark ({num_iterations} iterations)...")
    latencies = []
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []

    # Memory before
    try:
        import psutil
        ram_before = psutil.virtual_memory().used / 1e6  # MB
    except Exception:
        ram_before = 0.0

    t_bench_start = time.perf_counter()

    for i in range(num_iterations):
        frame = frames[i % len(frames)]
        result = model.predict(frame)
        latencies.append(result.total_latency_ms)
        preprocessing_times.append(result.preprocessing_time_ms)
        inference_times.append(result.inference_time_ms)
        postprocessing_times.append(result.postprocessing_time_ms)

    total_bench_time = time.perf_counter() - t_bench_start

    # Memory after
    try:
        ram_after = psutil.virtual_memory().used / 1e6  # MB
        ram_delta = ram_after - ram_before
    except Exception:
        ram_delta = 0.0

    latencies = np.array(latencies)
    benchmark_results = {
        "model": model_name,
        "device": device,
        "status": "SUCCESS",
        "num_iterations": num_iterations,
        "resolution": list(resolution),
        "latency_ms": {
            "mean":   round(float(np.mean(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "std":    round(float(np.std(latencies)), 2),
            "p50":    round(float(np.percentile(latencies, 50)), 2),
            "p90":    round(float(np.percentile(latencies, 90)), 2),
            "p95":    round(float(np.percentile(latencies, 95)), 2),
            "p99":    round(float(np.percentile(latencies, 99)), 2),
            "min":    round(float(np.min(latencies)), 2),
            "max":    round(float(np.max(latencies)), 2),
        },
        "throughput": {
            "fps_mean":  round(float(1000.0 / np.mean(latencies)), 1),
            "fps_p95":   round(float(1000.0 / np.percentile(latencies, 95)), 1),
            "fps_peak":  round(float(1000.0 / np.min(latencies)), 1),
        },
        "stage_breakdown_ms": {
            "preprocessing": round(float(np.mean(preprocessing_times)), 2),
            "inference":     round(float(np.mean(inference_times)), 2),
            "postprocessing": round(float(np.mean(postprocessing_times)), 2),
        },
        "memory_delta_mb": round(ram_delta, 1),
        "total_bench_time_s": round(total_bench_time, 2),
    }

    logger.info(
        f"  Results: "
        f"mean={benchmark_results['latency_ms']['mean']}ms | "
        f"p95={benchmark_results['latency_ms']['p95']}ms | "
        f"fps={benchmark_results['throughput']['fps_mean']}"
    )

    return benchmark_results


def run_benchmark_suite(
    models: Optional[List[str]] = None,
    device: str = "cpu",
    num_iterations: int = 100,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Run the complete benchmarking suite across multiple models.

    Args:
        models: List of model names to benchmark (None = all)
        device: Target device ('cpu' | 'cuda')
        num_iterations: Iterations per model
        output_path: Path to save JSON report

    Returns:
        Complete benchmark report
    """
    default_models = [
        "yolov8n",
        "yolov8s",
        "mobilenet_small",
        "mobilenet_large",
        "efficientnet_b0",
        "efficientnet_b1",
    ]
    models_to_bench = models or default_models

    logger.info("=" * 70)
    logger.info("EDGE AI VIDEO ANALYTICS — BENCHMARK SUITE")
    logger.info("=" * 70)

    # Collect device info
    device_info = get_device_info()
    logger.info("Device Information:")
    for k, v in device_info.items():
        logger.info(f"  {k}: {v}")
    logger.info("-" * 70)

    # Run benchmarks
    results = []
    for model_name in models_to_bench:
        result = benchmark_model(
            model_name=model_name,
            device=device,
            num_iterations=num_iterations,
        )
        results.append(result)
        logger.info(f"  ✓ {model_name} complete")
        logger.info("-" * 40)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'Avg Lat(ms)':<14} {'P95 Lat(ms)':<14} {'FPS':<10} {'Status'}")
    logger.info("-" * 70)
    for r in results:
        if r.get("status") == "SUCCESS":
            lat = r["latency_ms"]["mean"]
            p95 = r["latency_ms"]["p95"]
            fps = r["throughput"]["fps_mean"]
            logger.info(f"{r['model']:<20} {lat:<14.1f} {p95:<14.1f} {fps:<10.1f} ✓")
        else:
            logger.info(f"{r['model']:<20} {'—':<14} {'—':<14} {'—':<10} ✗ {r.get('error', '')}")
    logger.info("=" * 70)

    # Compile full report
    report = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_info": device_info,
        "benchmark_config": {
            "device": device,
            "num_iterations": num_iterations,
        },
        "results": results,
    }

    # Save report
    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nBenchmark report saved: {output_path}")

    return report


def main():
    """Command-line interface for the benchmarking tool."""
    parser = argparse.ArgumentParser(
        description="Edge AI Video Analytics — Performance Benchmarking Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all models on CPU
  python scripts/benchmark.py --device cpu

  # Benchmark YOLOv8n on CUDA with 200 iterations
  python scripts/benchmark.py --model yolov8n --device cuda --iterations 200

  # Full suite with JSON report
  python scripts/benchmark.py --device cuda --output benchmark_results.json
        """
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model to benchmark (None = all models)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Target device"
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON benchmark report"
    )

    args = parser.parse_args()

    models = [args.model] if args.model else None
    run_benchmark_suite(
        models=models,
        device=args.device,
        num_iterations=args.iterations,
        output_path=args.output or "benchmark_results.json",
    )


if __name__ == "__main__":
    main()
