# main.py - YOLO training pipeline
# Classes: Artillery, Missile, Radar, RocketLauncher, Soldier, Tank, Vehicle
#
# Full pipeline:
#   1. Prepare dataset   (calls dataset.py)
#   2. Train YOLO model
#   3. Validate
#   4. Export to ONNX (optional)
#
# Usage:
#   python main.py                                    # full pipeline with defaults
#   python main.py --data dataset/data.yaml           # skip dataset prep
#   python main.py --data dataset/data.yaml --model yolo11m.pt --epochs 150
#   python main.py --data dataset/data.yaml --device cpu --batch 8
#   python main.py --data dataset/data.yaml --export-onnx --no-validate

import os
# Prevent ultralytics from auto-installing the CPU onnxruntime
# when onnxruntime-gpu is already installed
os.environ["ULTRALYTICS_AUTO_INSTALL"] = "0"

import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Configuration defaults ────────────────────────────────────────────────────
DEFAULT_SOURCE_DIR  = "raw_dataset"   # raw images + YOLO labels (input to dataset.py)
DEFAULT_DATASET_DIR = "dataset"       # prepared dataset output (must match data.yaml path:)
DEFAULT_OUTPUT_DIR  = "results"       # parent folder for training run outputs
DEFAULT_RUN_NAME    = "detector"      # subfolder name inside DEFAULT_OUTPUT_DIR

# YOLO11 base model. Options from fastest/smallest to slowest/largest:
#   yolo11n.pt  yolo11s.pt  yolo11m.pt  yolo11l.pt  yolo11x.pt
# yolo11s is recommended for this task
DEFAULT_MODEL  = "yolo11s.pt"

DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ  = 640   # input resolution; must match the resolution used during export
DEFAULT_BATCH  = 16    # reduce to 8 if GPU runs out of memory
DEFAULT_DEVICE = 0     # 0 = first GPU; use "cpu" for CPU-only machines
# ─────────────────────────────────────────────────────────────────────────────


def train(
    data_yaml:  str,
    base_model: str = DEFAULT_MODEL,
    epochs:     int = DEFAULT_EPOCHS,
    imgsz:      int = DEFAULT_IMGSZ,
    batch:      int = DEFAULT_BATCH,
    device          = DEFAULT_DEVICE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    run_name:   str = DEFAULT_RUN_NAME,
) -> YOLO:
    """
    Fine-tune a YOLO11 model on the prepared dataset.

    Augmentation parameters are tuned for small object detection:
    - Moderate scale variance to simulate different drone altitudes
    - No vertical flip (preserves top-down orientation cues)
    - Low mixup to avoid blending visually similar classes (Tank/Vehicle)
    - High mosaic to expose the model to multi-object scenes

    Returns the loaded best-weights model after training completes.
    """
    print("\n" + "=" * 60)
    print("🚀  TRAINING")
    print("=" * 60)
    print(f"  Base model : {base_model}")
    print(f"  Data       : {data_yaml}")
    print(f"  Epochs     : {epochs}")
    print(f"  Image size : {imgsz}")
    print(f"  Batch size : {batch}")
    print(f"  Device     : {device}")
    print(f"  Output     : {output_dir}/{run_name}")

    model = YOLO(base_model)

    model.train(
        data    = data_yaml,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = output_dir,
        name    = run_name,

        # Early stopping: halt if validation mAP does not improve for this many epochs
        patience = 20,

        # ── Augmentation ──────────────────────────────────────────────────────
        hsv_h     = 0.015,  # small hue shift — preserve military colour signatures
        hsv_s     = 0.7,    # saturation variance for different lighting conditions
        hsv_v     = 0.4,    # brightness variance (day/overcast/shadow)
        degrees   = 5,      # slight rotation — drone movement during turns
        translate = 0.1,    # positional jitter
        scale     = 0.5,    # scale variance to simulate altitude differences
        flipud    = 0.0,    # no vertical flip — top-down orientation is meaningful
        fliplr    = 0.5,    # horizontal flip is fine
        mosaic    = 1.0,    # mosaic critical for multi-object scenes
        mixup     = 0.05,   # low mixup — avoid blending visually similar classes

        # ── Optimiser ─────────────────────────────────────────────────────────
        optimizer    = "auto",   # ultralytics selects best optimiser for the model
        lr0          = 0.01,     # initial learning rate
        lrf          = 0.01,     # final learning rate
        momentum     = 0.937,
        weight_decay = 0.0005,   # L2 regularisation to prevent overfitting
        warmup_epochs= 3,        # gradual LR ramp-up at the start of training

        # ── Misc ──────────────────────────────────────────────────────────────
        amp     = True,   # Automatic Mixed Precision — faster training on CUDA
        plots   = True,   # save loss/metric curves to results/run_name/
        verbose = True,
        save    = True,
    )

    best_pt = Path(output_dir) / run_name / "weights" / "best.pt"
    print(f"\n✅  Training complete. Best weights: {best_pt}")
    return YOLO(str(best_pt))


def validate(model: YOLO) -> None:
    """
    Run the built-in Ultralytics validation on the model's configured dataset.
    Reports mAP50, mAP50-95, Precision, and Recall.
    """
    print("\n" + "=" * 60)
    print("🔍  VALIDATION")
    print("=" * 60)

    metrics = model.val()

    print("\n📊  Results:")
    print(f"    mAP50      : {metrics.box.map50:.4f}")
    print(f"    mAP50-95   : {metrics.box.map:.4f}")
    print(f"    Precision  : {metrics.box.mp:.4f}")
    print(f"    Recall     : {metrics.box.mr:.4f}")


def export_onnx(model: YOLO, imgsz: int = DEFAULT_IMGSZ) -> None:
    """
    Export the trained model to ONNX format.
    half=True enables FP16 (requires CUDA at export time).
    simplify=True applies ONNX graph optimisations for faster inference.
    The exported .onnx file is written next to the source .pt file.
    """
    print("\n" + "=" * 60)
    print("📦  ONNX EXPORT")
    print("=" * 60)

    model.export(format="onnx", imgsz=imgsz, half=True, dynamic=False, simplify=True)
    print("✅  Exported to ONNX.")
    print("    Copy the .onnx file to your deployment directory.")
    print("    Select 'ONNX' as the backend in app.py.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLO military object detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    g = parser.add_argument_group("Dataset")
    g.add_argument("--source",         default=DEFAULT_SOURCE_DIR,
                   help="Raw dataset folder (images + YOLO .txt labels)")
    g.add_argument("--output-dataset", default=DEFAULT_DATASET_DIR,
                   help="Prepared dataset output folder")
    g.add_argument("--classes",        nargs="+", default=None,
                   help="Class names in index order. Defaults to: "
                        "Artillery Missile Radar RocketLauncher Soldier Tank Vehicle")
    g.add_argument("--skip-prep",      action="store_true",
                   help="Skip dataset preparation — requires --data to point to existing data.yaml")
    g.add_argument("--data",           default=None,
                   help="Path to an existing data.yaml (automatically implies --skip-prep)")

    # Training arguments
    g = parser.add_argument_group("Training")
    g.add_argument("--model",   default=DEFAULT_MODEL,
                   help="YOLO base model to fine-tune (n/s/m/l/x suffix)")
    g.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    g.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ,
                   help="Input image size (square). Must match export imgsz.")
    g.add_argument("--batch",   type=int, default=DEFAULT_BATCH,
                   help="Batch size. Reduce to 8 if GPU runs out of memory.")
    g.add_argument("--device",  default=str(DEFAULT_DEVICE),
                   help="Training device: 0 (first GPU), 0,1 (multi-GPU), or cpu")
    g.add_argument("--project", default=DEFAULT_OUTPUT_DIR,
                   help="Parent folder for training output")
    g.add_argument("--name",    default=DEFAULT_RUN_NAME,
                   help="Run subfolder name. Use distinct names to avoid overwriting previous runs.")

    # Post-training arguments
    g = parser.add_argument_group("Post-training")
    g.add_argument("--no-validate", action="store_true",
                   help="Skip validation step after training")
    g.add_argument("--export-onnx", action="store_true",
                   help="Automatically export best.pt to ONNX after training")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯  MILITARY OBJECT DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Dataset preparation ───────────────────────────────────────────
    data_yaml = args.data

    if data_yaml and os.path.exists(data_yaml):
        print(f"\n✅  Using existing data.yaml: {data_yaml}")
    else:
        if args.skip_prep:
            print("\n❌  --skip-prep requires a valid --data path. Exiting.")
            return

        # Import here so dataset.py can also be run as a standalone script
        from dataset import prepare_dataset, CLASS_NAMES

        class_names = args.classes if args.classes else CLASS_NAMES

        data_yaml = prepare_dataset(
            source_dir  = args.source,
            output_dir  = args.output_dataset,
            class_names = class_names,
        )

        if data_yaml is None:
            print("\n❌  Dataset preparation failed. Exiting.")
            return

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    # Convert device to int if numeric (CUDA device index), else keep as string ("cpu")
    device = int(args.device) if args.device.isdigit() else args.device

    trained_model = train(
        data_yaml  = data_yaml,
        base_model = args.model,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        device     = device,
        output_dir = args.project,
        run_name   = args.name,
    )

    # ── Step 3: Validate ──────────────────────────────────────────────────────
    if not args.no_validate:
        validate(trained_model)

    # ── Step 4: ONNX export ───────────────────────────────────────────────────
    if args.export_onnx:
        export_onnx(trained_model, imgsz=args.imgsz)
    else:
        ans = input("\n📦  Export to ONNX now? (y/n): ").strip().lower()
        if ans == "y":
            export_onnx(trained_model, imgsz=args.imgsz)

    # ── Summary ───────────────────────────────────────────────────────────────
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("✅  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Trained weights : {best_pt}")
    print(f"  Classes         : Artillery, Missile, Radar, RocketLauncher, Soldier, Tank, Vehicle")
    print(f"\n  Next steps:")
    print(f"    1. Copy best.pt (and best.onnx if exported) to your app directory")
    print(f"    2. Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()