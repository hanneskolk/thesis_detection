# app.py
# Detects: Artillery, Missile, Radar, RocketLauncher, Soldier, Tank, Vehicle
# Backends (priority order): TensorRT (.engine) → ONNX (.onnx) → PyTorch (.pt)
#
# Usage:
#   streamlit run app.py
#   streamlit run app.py --server.address 0.0.0.0 --server.port 8501 if used on TensorDock VM

import os
# Must be set before ultralytics is imported to prevent auto-installing CPU onnxruntime
os.environ["ULTRALYTICS_AUTO_INSTALL"] = "0"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import json
import random
from pathlib import Path

# ── Default model paths (editable at runtime via sidebar) ─────────────────────
PT_MODEL_PATH   = "best.pt"
TRT_MODEL_PATH  = "best.engine"
ONNX_MODEL_PATH = "best.onnx"
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Military Object Detection",
    page_icon="🎯",
    layout="wide",
)


# ══════════════════════════════════════════════════════════════════════════════
# Export helpers
# ══════════════════════════════════════════════════════════════════════════════

def export_to_tensorrt(pt_path: str, engine_path: str, imgsz: int = 640) -> bool:
    """
    Convert best.pt to best.engine using TensorRT FP16 compilation.
    Requires: NVIDIA GPU, CUDA toolkit, tensorrt Python package.
    The compiled engine is hardware-specific.
    Returns True on success.
    """
    try:
        with st.spinner("⚙️ Exporting to TensorRT — this may take several minutes…"):
            m = YOLO(pt_path)
            # half=True enables FP16 for best speed/accuracy tradeoff on NVIDIA GPUs
            m.export(format="engine", imgsz=imgsz, half=True)
        # Ultralytics writes the .engine next to the .pt file — move it to engine_path
        auto = pt_path.replace(".pt", ".engine")
        if os.path.exists(auto) and auto != engine_path:
            os.rename(auto, engine_path)
        return os.path.exists(engine_path)
    except Exception as e:
        st.error(f"TensorRT export failed: {e}")
        return False


def export_to_onnx(pt_path: str, onnx_path: str, imgsz: int = 640) -> bool:
    """
    Convert best.pt to best.onnx.
    ONNX is portable across machines.
    simplify=True applies graph optimisations that improve inference speed.
    Returns True on success.
    """
    try:
        with st.spinner("⚙️ Exporting to ONNX…"):
            m = YOLO(pt_path)
            # half=True requires CUDA; dynamic=False produces a fixed-input-size graph
            # which is faster for fixed 640×640 inference
            m.export(format="onnx", imgsz=imgsz, half=True, dynamic=False, simplify=True)
        auto = pt_path.replace(".pt", ".onnx")
        if os.path.exists(auto) and auto != onnx_path:
            os.rename(auto, onnx_path)
        return os.path.exists(onnx_path)
    except Exception as e:
        st.error(f"ONNX export failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Model loader — TRT → ONNX → PT cascade
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(pt_path: str, trt_path: str, onnx_path: str, prefer: str) -> tuple:
    """
    Load the YOLO model using the preferred backend.
    Falls back automatically if the preferred backend fails or is unavailable:
      TensorRT → ONNX → PyTorch

    Args:
        pt_path:   Path to PyTorch weights (.pt)
        trt_path:  Path to TensorRT engine (.engine)
        onnx_path: Path to ONNX model (.onnx)
        prefer:    One of "tensorrt", "onnx", "pytorch"

    Returns:
        (model, backend_label) — backend_label describes the runtime in use.
        Return (None, "") if no backend could be loaded.
    """
    if prefer == "tensorrt":
        if os.path.exists(trt_path):
            try:
                return YOLO(trt_path), "TensorRT (.engine)"
            except Exception as e:
                st.warning(f"⚠️ TensorRT failed ({e}) — trying ONNX.")
        else:
            st.warning("⚠️ No .engine file found — trying ONNX.")

        if os.path.exists(onnx_path):
            try:
                return YOLO(onnx_path), "ONNX (.onnx)"
            except Exception as e:
                st.warning(f"⚠️ ONNX failed ({e}) — falling back to PyTorch.")
        else:
            st.warning("⚠️ No .onnx file found — falling back to PyTorch.")

    elif prefer == "onnx":
        if os.path.exists(onnx_path):
            try:
                return YOLO(onnx_path), "ONNX (.onnx)"
            except Exception as e:
                st.warning(f"⚠️ ONNX failed ({e}) — falling back to PyTorch.")
        else:
            st.warning("⚠️ No .onnx file found — falling back to PyTorch.")

    # PyTorch fallback — always available if best.pt exists
    if not os.path.exists(pt_path):
        st.error(f"❌ Model not found at `{pt_path}`. Place best.pt in the project root.")
        return None, ""
    return YOLO(pt_path), "PyTorch (.pt)"


# ══════════════════════════════════════════════════════════════════════════════
# Overlay helper
# ══════════════════════════════════════════════════════════════════════════════

def draw_stats_overlay(img: np.ndarray, boxes, inference_ms: float,
                       backend: str, track_ids: list = None) -> np.ndarray:
    """
    Composites a semi-transparent statistics panel onto the annotated frame.
    Shows: active backend, object count, mean confidence, inference time,
    and (if tracking) the number of unique tracked IDs in this frame.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (340, 175), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "DETECTION STATUS", (20, 38), font, 0.65, (255, 255, 255), 2)

    # Badge colour: cyan = TensorRT, green = ONNX, grey = PyTorch
    badge_color = (
        (0, 200, 255) if "TensorRT" in backend else
        (100, 255, 100) if "ONNX" in backend else
        (180, 180, 180)
    )
    cv2.putText(img, backend, (20, 62), font, 0.50, badge_color, 1)

    y = 90
    if len(boxes) > 0:
        color = (0, 255, 80)
        cv2.putText(img, f"Objects:   {len(boxes)}", (20, y),        font, 0.65, color, 2)
        cv2.putText(img, f"Avg conf:  {float(boxes.conf.mean()):.1%}", (20, y + 28), font, 0.65, color, 2)
        cv2.putText(img, f"Inference: {inference_ms:.1f} ms",         (20, y + 56), font, 0.65, color, 2)
        if track_ids is not None:
            cv2.putText(img, f"Track IDs: {len(set(track_ids))}", (20, y + 84), font, 0.65, (0, 210, 255), 2)
    else:
        color = (160, 160, 160)
        cv2.putText(img, "Objects: 0  —  CLEAR", (20, y),       font, 0.65, color, 2)
        cv2.putText(img, f"Inference: {inference_ms:.1f} ms",   (20, y + 28), font, 0.65, color, 2)

    return img


# ══════════════════════════════════════════════════════════════════════════════
# Validation helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """
    Parse a YOLO-format .txt label file and convert normalised coordinates
    to absolute pixel coordinates.

    YOLO format per line: <class_id> <cx> <cy> <w> <h>  (all normalised [0,1])

    Returns a list of dicts with keys: cls, x1, y1, x2, y2
    Returns an empty list if the label file does not exist.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append({"cls": cls_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def iou(a: dict, b: dict) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.
    Both inputs must have keys: x1, y1, x2, y2 (absolute pixel coordinates).
    Returns a value in [0, 1]; returns 0.0 if union is zero.
    """
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_sample_metrics(gt_boxes: list, pred_boxes: list,
                            iou_thresh: float = 0.5) -> dict:
    """
    Compute TP, FP, FN, Precision, Recall, F1, and mean IoU for a single image.

    Matching strategy: each prediction is matched to the highest-IoU
    unmatched ground truth box of the same class that meets the IoU threshold.
    Unmatched predictions = FP; unmatched GT boxes = FN.

    mean_iou is the average IoU across matched (TP) pairs only.
    """
    matched_gt = set()
    tp = fp = 0
    ious = []

    for pred in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if gt["cls"] != pred["cls"]:
                continue  # only match same class
            v = iou(gt, pred)
            if v > best_iou:
                best_iou, best_idx = v, i

        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def draw_gt_pred(img: np.ndarray, gt_boxes: list, pred_boxes: list,
                 names: dict) -> np.ndarray:
    """
    Draw ground truth boxes (green, labelled GT:class) and prediction boxes
    (blue, labelled PR:class confidence%) onto a copy of the image.
    Used in the validation tab sample image grid.
    """
    out = img.copy()
    for b in gt_boxes:
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
        label = names.get(b["cls"], str(b["cls"]))
        cv2.putText(out, f"GT:{label}", (b["x1"], max(b["y1"] - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    for b in pred_boxes:
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 100, 255), 2)
        label = names.get(b["cls"], str(b["cls"]))
        cv2.putText(out, f"PR:{label} {b['conf']:.0%}",
                    (b["x1"], b["y2"] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
    return out


def scan_dataset_split(root: Path, split: str) -> list[tuple[Path, Path]]:
    """
    Return all (image_path, label_path) pairs for the given dataset split.
    Expects structure: root/split/images/*.jpg  and  root/split/labels/*.txt
    Label path is included even if the file does not exist — parse_yolo_label
    handles missing files gracefully by returning an empty list.
    """
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    if not img_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        pairs.append((img_path, lbl_path))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — global settings and backend management
# ══════════════════════════════════════════════════════════════════════════════

st.title("🎯 Military Object Detection")
st.markdown(
    "Detects **Artillery, Missile, Radar, Rocket Launchers, Soldiers, Tanks, and Vehicles** "
    "using your trained YOLO model."
)
st.markdown("---")

st.sidebar.title("⚙️ Settings")

# Model path inputs — allow runtime override without restarting the app
pt_path_input   = st.sidebar.text_input("PyTorch model (.pt)",      value=PT_MODEL_PATH)
trt_path_input  = st.sidebar.text_input("TensorRT model (.engine)", value=TRT_MODEL_PATH)
onnx_path_input = st.sidebar.text_input("ONNX model (.onnx)",       value=ONNX_MODEL_PATH)

confidence = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, 0.40, 0.05,
    help="Minimum detection score to display. Lower = more detections, higher = fewer false positives."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Backend")

backend_pref = st.sidebar.radio(
    "Preferred backend",
    options=["pytorch", "onnx", "tensorrt"],
    format_func=lambda x: {
        "pytorch":   "PyTorch (.pt)",
        "onnx":      "ONNX (.onnx)",
        "tensorrt":  "TensorRT (.engine)",
    }[x],
    index=0,
    help="The app falls back automatically if the preferred backend is unavailable.",
)

# Export buttons side by side
col_trt, col_onnx = st.sidebar.columns(2)

with col_trt:
    if st.button("Export TRT", help="Requires NVIDIA GPU + TensorRT. Engine is device-specific."):
        if not os.path.exists(pt_path_input):
            st.sidebar.error("❌ .pt file not found.")
        else:
            ok = export_to_tensorrt(pt_path_input, trt_path_input)
            if ok:
                st.sidebar.success("✅ TRT export complete.")
                st.cache_resource.clear()
            else:
                st.sidebar.error("TRT export failed. Check GPU + TensorRT installation.")

with col_onnx:
    if st.button("Export ONNX", help="Portable across machines. Recommended for cloud VMs."):
        if not os.path.exists(pt_path_input):
            st.sidebar.error("❌ .pt file not found.")
        else:
            ok = export_to_onnx(pt_path_input, onnx_path_input)
            if ok:
                st.sidebar.success("✅ ONNX export complete.")
                st.cache_resource.clear()
            else:
                st.sidebar.error("ONNX export failed.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "**TensorRT** — fastest, NVIDIA GPU required. "
    "Engine must be rebuilt on each GPU model.\n\n"
    "**ONNX** — ~1.5–2× faster than PyTorch, portable, no extra drivers needed."
)

# Load model (cached — changing path or backend requires a rerun)
model, backend = load_model(pt_path_input, trt_path_input, onnx_path_input, backend_pref)
if model is None:
    st.stop()

st.sidebar.success(f"Active backend: **{backend}**")


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_img, tab_vid, tab_val = st.tabs(["📷 Image", "🎥 Video", "📊 Validation"])


# ── Image detection tab ───────────────────────────────────────────────────────
with tab_img:
    st.header("Image Detection")
    uploaded = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="img_upload",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("🔍 Running detection…"):
            results   = model(image, conf=confidence)
            annotated = results[0].plot(labels=False, conf=False)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB) # YOLO returns BGR; Streamlit needs RGB
            boxes     = results[0].boxes
            ms        = results[0].speed["inference"]
            annotated = draw_stats_overlay(annotated, boxes, ms, backend)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Detections")
            st.image(annotated, use_container_width=True)

        if len(boxes) > 0:
            st.markdown("---")
            st.subheader("🎯 Detection Details")
            names = model.names
            for i, box in enumerate(boxes):
                conf_val = float(box.conf[0])
                cls_id   = int(box.cls[0])
                cls_name = names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                st.write(
                    f"**{cls_name.capitalize()} {i+1}:** "
                    f"conf = {conf_val:.2%}  |  "
                    f"bbox = [{x1}, {y1}, {x2}, {y2}]"
                )
        else:
            st.info("✅ No objects detected in this image.")


# ── Video detection tab ───────────────────────────────────────────────────────
with tab_vid:
    st.header("Video Detection")
    st.info("ℹ️ If the tab switches after upload, click **Video** again.")

    video_file = st.file_uploader(
        "Upload a video (MP4 / AVI / MOV / MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )
    enable_tracking = st.checkbox(
        "Enable object tracking",
        value=True,
        help=(
            "Uses ByteTrack to assign persistent IDs to objects across frames. "
            "Draws trajectory trails and counts unique objects over the full video."
        ),
    )

    if video_file and st.button("▶️ Process Video"):
        # Write upload to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()

        cap          = cv2.VideoCapture(tfile.name)
        fps          = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)  # guard against 0 FPS
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Codec selection
        def open_writer(path: str, fourcc_str: str) -> cv2.VideoWriter:
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height))
            return w if w.isOpened() else None

        raw_path = tempfile.mktemp(suffix=".mp4")
        out = (
            open_writer(raw_path, "mp4v") or   # software MPEG-4 — always available
            open_writer(raw_path, "XVID") or   # fallback
            open_writer(raw_path, "MJPG")      # last resort (larger file)
        )

        if out is None:
            st.error("❌ Could not open any video encoder. Check your OpenCV installation.")
            cap.release()
            st.stop()

        output_path = raw_path
        progress    = st.progress(0)
        status_text = st.empty()
        frame_count = 0
        unique_ids: set     = set()
        track_history: dict = {}

        with st.spinner("Processing video…"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if enable_tracking:
                    # persist=True maintains tracker state between frames
                    results = model.track(frame, conf=confidence, persist=True, verbose=False)
                else:
                    results = model(frame, conf=confidence, verbose=False)

                # Plot bounding boxes without text labels
                annotated_frame = results[0].plot(labels=False, conf=False)

                # Draw trajectory trails when tracking is active
                if enable_tracking and results[0].boxes.id is not None:
                    boxes_xywh = results[0].boxes.xywh.cpu()
                    ids        = results[0].boxes.id.int().cpu().tolist()

                    for box, tid in zip(boxes_xywh, ids):
                        cx, cy = float(box[0]), float(box[1])
                        track_history.setdefault(tid, []).append((cx, cy))

                        # Keep only the last 50 points to avoid memory growth on long videos
                        if len(track_history[tid]) > 50:
                            track_history[tid].pop(0)

                        pts = np.array(track_history[tid], dtype=np.int32).reshape(-1, 1, 2)
                        if len(pts) > 1:
                            cv2.polylines(annotated_frame, [pts], False, (0, 0, 0), 5)       # shadow
                            cv2.polylines(annotated_frame, [pts], False, (0, 230, 255), 3)   # trail

                    unique_ids.update(ids)

                out.write(annotated_frame)
                frame_count += 1
                progress.progress(min(frame_count / max(total_frames, 1), 1.0))
                status_text.text(f"Frame {frame_count} / {total_frames}  |  {backend}")

        cap.release()
        out.release()
        # clean up the uploaded temp file
        os.unlink(tfile.name)
        status_text.empty()

        st.success(f"✅ Video processed with **{backend}**!")

        if enable_tracking and unique_ids:
            st.info(
                f"🎯 Tracked **{len(unique_ids)}** unique object(s) "
                f"across {frame_count} frames."
            )

        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "📥 Download annotated video",
                f,
                "detection_output.mp4",
                "video/mp4",
            )


# ── Validation tab ────────────────────────────────────────────────────────────
with tab_val:
    st.header("📊 Model Validation")
    st.markdown(
        "Point to your dataset root (the folder containing `train/`, `valid/`, `test/` subfolders). "
        "The app runs inference on a sample of images, compares predictions against YOLO labels, "
        "and reports Precision, Recall, F1, and mean IoU."
    )

    dataset_root = st.text_input(
        "Dataset root path", value="dataset",
        help="Folder containing train/, valid/, test/ subfolders with images/ and labels/ inside each.",
    )

    val_col1, val_col2, val_col3 = st.columns(3)
    with val_col1:
        split_choice = st.selectbox("Split to evaluate", ["valid", "test", "train"])
    with val_col2:
        n_samples = st.number_input(
            "Max images to sample", min_value=1, max_value=500, value=50,
            help="Larger samples give more reliable metrics but take longer to run.",
        )
    with val_col3:
        iou_thresh = st.slider(
            "IoU threshold", 0.1, 0.9, 0.5, 0.05,
            help="Minimum overlap between predicted and ground truth box to count as a true positive.",
        )

    show_samples = st.checkbox("Show annotated sample images", value=True)
    n_display    = st.slider("Images to display", 1, 20, 6, disabled=not show_samples)

    if st.button("▶️ Run Validation"):
        root = Path(dataset_root)
        if not root.exists():
            st.error(f"❌ Path not found: `{dataset_root}`")
            st.stop()

        pairs = scan_dataset_split(root, split_choice)
        if not pairs:
            st.error(f"❌ No images found in `{dataset_root}/{split_choice}/images/`")
            st.stop()

        # Fixed seed for reproducible sampling — same images selected on repeated runs
        random.seed(42)
        sampled = random.sample(pairs, min(n_samples, len(pairs)))
        st.info(
            f"Evaluating **{len(sampled)}** images from `{split_choice}/` split  |  "
            f"backend: **{backend}**"
        )

        names = model.names  # {0: 'Artillery', ...}

        # Accumulators for aggregate metrics
        all_tp = all_fp = all_fn = 0
        all_ious: list[float] = []
        per_class_fn: dict[int, int] = {}
        per_class_fp: dict[int, int] = {}
        sample_visuals: list = []    # list of (annotated_img, filename, per_image_metrics)

        progress = st.progress(0)

        for idx, (img_path, lbl_path) in enumerate(sampled):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]

            # Load ground truth boxes from YOLO label file
            gt_boxes = parse_yolo_label(lbl_path, w, h)

            # Run model inference
            results    = model(img_bgr, conf=confidence, verbose=False)
            pred_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pred_boxes.append({
                    "cls":  int(box.cls[0]),
                    "conf": float(box.conf[0]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

            # Per-image metrics
            m = compute_sample_metrics(gt_boxes, pred_boxes, iou_thresh)
            all_tp += m["tp"]
            all_fp += m["fp"]
            all_fn += m["fn"]
            if m["mean_iou"] > 0:
                all_ious.append(m["mean_iou"])

            # Accumulate raw counts per class for the breakdown table
            for b in gt_boxes:
                c = b["cls"]
                per_class_fn[c] = per_class_fn.get(c, 0) + 1
            for b in pred_boxes:
                c = b["cls"]
                per_class_fp[c] = per_class_fp.get(c, 0) + 1

            # Collecting visuals for the image grid
            if show_samples and len(sample_visuals) < n_display:
                vis = draw_gt_pred(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    gt_boxes, pred_boxes, names,
                )
                sample_visuals.append((vis, img_path.name, m))

            progress.progress((idx + 1) / len(sampled))

        progress.empty()

        # Aggregate metrics
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        mean_iou  = float(np.mean(all_ious)) if all_ious else 0.0

        st.markdown("---")
        st.subheader("Overall Metrics")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Precision",          f"{precision:.3f}")
        m2.metric("Recall",             f"{recall:.3f}")
        m3.metric("F1 Score",           f"{f1:.3f}")
        m4.metric("Mean IoU",           f"{mean_iou:.3f}")
        m5.metric("True Positives",     all_tp)
        m6.metric("FP / FN",            f"{all_fp} / {all_fn}")

        # Per-class breakdown
        st.markdown("---")
        st.subheader("Per-class Breakdown")

        all_cls = set(per_class_fn.keys()) | set(per_class_fp.keys())
        if all_cls:
            rows = []
            for c in sorted(all_cls):
                cls_name   = names.get(c, f"class_{c}")
                gt_count   = per_class_fn.get(c, 0)
                pred_count = per_class_fp.get(c, 0)
                rows.append({
                    "Class":        cls_name,
                    "GT objects":   gt_count,
                    "Predictions":  pred_count,
                    "Δ (pred−GT)":  pred_count - gt_count,
                })
            st.table(rows)
        else:
            st.info("No detections or ground truth found in this sample.")

        # Confidence distribution histogram
        st.markdown("---")
        st.subheader("Confidence Distribution")

        all_confs: list[float] = []
        for img_path, lbl_path in sampled:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            res = model(img_bgr, conf=confidence, verbose=False)
            for box in res[0].boxes:
                all_confs.append(float(box.conf[0]))

        if all_confs:
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            hist, _ = np.histogram(all_confs, bins=bins)
            hist_data = {
                f"{int(bins[i]*100)}–{int(bins[i+1]*100)}%": int(hist[i])
                for i in range(len(hist))
            }
            st.bar_chart(hist_data)
            st.caption(
                f"Total detections: **{len(all_confs)}**  |  "
                f"Mean confidence: **{np.mean(all_confs):.1%}**  |  "
                f"Median: **{np.median(all_confs):.1%}**"
            )
        else:
            st.info("No detections above the confidence threshold in this sample.")

        # Sample image grid — GT boxes (green) vs predictions (blue)
        if show_samples and sample_visuals:
            st.markdown("---")
            st.subheader("Sample Predictions vs Ground Truth")
            st.caption("🟩 Green = ground truth  |  🟦 Blue = model prediction")

            cols_per_row = 3
            for row_start in range(0, len(sample_visuals), cols_per_row):
                row_imgs = sample_visuals[row_start: row_start + cols_per_row]
                cols = st.columns(len(row_imgs))
                for col, (vis_img, fname, img_m) in zip(cols, row_imgs):
                    with col:
                        st.image(vis_img, use_container_width=True)
                        st.caption(
                            f"**{fname}**\n"
                            f"P={img_m['precision']:.2f}  R={img_m['recall']:.2f}  "
                            f"F1={img_m['f1']:.2f}  IoU={img_m['mean_iou']:.2f}\n"
                            f"TP={img_m['tp']}  FP={img_m['fp']}  FN={img_m['fn']}"
                        )

        # JSON export for cross-run comparison
        st.markdown("---")
        export_data = {
            "backend":              backend,
            "split":                split_choice,
            "n_samples":            len(sampled),
            "iou_threshold":        iou_thresh,
            "confidence_threshold": confidence,
            "overall": {
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "mean_iou":  round(mean_iou, 4),
                "tp": all_tp, "fp": all_fp, "fn": all_fn,
            },
        }
        st.download_button(
            "📥 Download validation results (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name="validation_results.json",
            mime="application/json",
        )


if __name__ == "__main__":
    pass