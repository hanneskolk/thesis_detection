# benchmark.py - Inference backend benchmark
# Measures per-image latency for PyTorch, ONNX, and TensorRT backends
#
# Usage:
#   python benchmark.py                                      # uses defaults
#   python benchmark.py --images dataset/test/images        # custom image dir
#   python benchmark.py --n 200 --warmup 20                 # more samples
#   python benchmark.py --pt best.pt --onnx best.onnx       # custom model paths
#   python benchmark.py --no-trt                            # skip TensorRT

import argparse
import time
import numpy as np
import cv2
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_IMAGE_DIR = "dataset/test/images"
DEFAULT_PT        = "best.pt"
DEFAULT_ONNX      = "best.onnx"
DEFAULT_TRT       = "best.engine"
DEFAULT_N         = 100     # images to benchmark per backend
DEFAULT_WARMUP    = 10      # warmup runs before timing starts (not included in results)
DEFAULT_CONF      = 0.40
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(image_dir: str, n: int) -> list[Path]:
    """Collect up to n image paths from the given directory."""
    p = Path(image_dir)
    if not p.exists():
        return []

    images = [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMG_EXTS]

    if not images:
        return []

    # Repeat list if fewer images than requested, so warmup + benchmark always runs
    while len(images) < n + DEFAULT_WARMUP:
        images = images * 2

    return images[:n + DEFAULT_WARMUP]   # first DEFAULT_WARMUP are warmup-only


def benchmark_backend(name: str, model_path: str, images: list[Path],
                       n: int, warmup: int, conf: float) -> dict | None:
    """
    Run inference on all images and return timing statistics.
    Returns None if the backend is unavailable or fails to load.
    """
    from ultralytics import YOLO

    if not Path(model_path).exists():
        print(f"  ⚠️  Skipping {name} — file not found: {model_path}")
        return None

    print(f"\n  Loading {name} from {model_path} ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"  ❌ Failed to load {name}: {e}")
        return None

    bench_images = images[:warmup + n]

    print(f"  Warming up ({warmup} frames) ...")
    for img_path in bench_images[:warmup]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        model(frame, conf=conf, verbose=False)

    print(f"  Benchmarking ({n} frames) ...")
    times = []
    for img_path in bench_images[warmup:]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        t0 = time.perf_counter()
        model(frame, conf=conf, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    if not times:
        print(f"  ❌ No frames were processed for {name}.")
        return None

    times_arr = np.array(times)
    stats = {
        "backend":  name,
        "n":        len(times),
        "mean_ms":  float(np.mean(times_arr)),
        "std_ms":   float(np.std(times_arr)),
        "min_ms":   float(np.min(times_arr)),
        "max_ms":   float(np.max(times_arr)),
        "p50_ms":   float(np.percentile(times_arr, 50)),
        "p95_ms":   float(np.percentile(times_arr, 95)),
        "fps":      1000.0 / float(np.mean(times_arr)),
    }

    print(f"  ✅ {name}: mean={stats['mean_ms']:.1f}ms  "
          f"std={stats['std_ms']:.1f}ms  "
          f"p50={stats['p50_ms']:.1f}ms  "
          f"p95={stats['p95_ms']:.1f}ms  "
          f"FPS={stats['fps']:.1f}")

    return stats


def print_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    if not results:
        return

    header = f"\n{'Backend':<18} {'N':>5} {'Mean (ms)':>10} {'Std':>8} {'P50':>8} {'P95':>8} {'FPS':>8} {'Speedup':>9}"
    print("\n" + "=" * len(header))
    print("BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Use PyTorch mean as the baseline for speedup calculation
    baseline = next((r["mean_ms"] for r in results if "PyTorch" in r["backend"]), None)

    for r in results:
        speedup = f"{baseline / r['mean_ms']:.2f}×" if baseline else "—"
        if "PyTorch" in r["backend"]:
            speedup = "1.00×  (baseline)"
        print(
            f"{r['backend']:<18} {r['n']:>5} {r['mean_ms']:>10.1f} "
            f"{r['std_ms']:>8.1f} {r['p50_ms']:>8.1f} {r['p95_ms']:>8.1f} "
            f"{r['fps']:>8.1f} {speedup:>9}"
        )

    print("=" * len(header))
    print("All times in milliseconds. Speedup relative to PyTorch baseline.")
    print(f"Real-time threshold (25 FPS) = 40.0 ms/frame\n")


def save_csv(results: list[dict], out_path: str) -> None:
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"✅  Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency across PyTorch, ONNX, and TensorRT backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images",  default=DEFAULT_IMAGE_DIR, help="Directory of test images")
    parser.add_argument("--n",       type=int, default=DEFAULT_N,     help="Frames to benchmark per backend")
    parser.add_argument("--warmup",  type=int, default=DEFAULT_WARMUP, help="Warmup frames (excluded from results)")
    parser.add_argument("--conf",    type=float, default=DEFAULT_CONF, help="Detection confidence threshold")
    parser.add_argument("--pt",      default=DEFAULT_PT,   help="PyTorch model path")
    parser.add_argument("--onnx",    default=DEFAULT_ONNX, help="ONNX model path")
    parser.add_argument("--trt",     default=DEFAULT_TRT,  help="TensorRT model path")
    parser.add_argument("--no-pt",   action="store_true",  help="Skip PyTorch")
    parser.add_argument("--no-onnx", action="store_true",  help="Skip ONNX")
    parser.add_argument("--no-trt",  action="store_true",  help="Skip TensorRT")
    parser.add_argument("--out",     default="benchmark_results.csv", help="CSV output path")
    args = parser.parse_args()

    print("=" * 60)
    print("⚡ INFERENCE BACKEND BENCHMARK")
    print("=" * 60)

    # ── Collect images ────────────────────────────────────────────────────────
    images = collect_images(args.images, args.n + args.warmup)

    if not images:
        print(f"\n❌  No images found in: {args.images}")
        print("    Check the path exists and contains .jpg/.png files.")
        print("    Example: python benchmark.py --images dataset/test/images")
        return

    print(f"\n✅  Found images in: {args.images}")
    print(f"    Using {min(args.n, len(images))} for benchmarking + {args.warmup} warmup frames")
    print(f"    Confidence threshold: {args.conf}")

    # ── Run benchmarks ────────────────────────────────────────────────────────
    backends = []
    if not args.no_pt:
        backends.append(("PyTorch (.pt)",      args.pt))
    if not args.no_onnx:
        backends.append(("ONNX (.onnx)",       args.onnx))
    if not args.no_trt:
        backends.append(("TensorRT (.engine)", args.trt))

    results = []
    for name, path in backends:
        print(f"\n{'─' * 40}")
        print(f"Backend: {name}")
        stats = benchmark_backend(name, path, images, args.n, args.warmup, args.conf)
        if stats:
            results.append(stats)

    # ── Output ────────────────────────────────────────────────────────────────
    if not results:
        print("\n❌  No backends produced results.")
        return

    print_table(results)
    save_csv(results, args.out)


if __name__ == "__main__":
    main()