from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import streamlit as st

try:
    from ultralytics import YOLO
except Exception as import_error:  # pragma: no cover - handled at runtime
    YOLO = None
    IMPORT_ERROR = import_error
else:
    IMPORT_ERROR = None


st.set_page_config(page_title="Florodye Cell Detection", page_icon="🔬", layout="wide")


BASE_DIR = Path(__file__).resolve().parent


def find_model_path() -> Path | None:
    search_roots = [BASE_DIR, BASE_DIR / "models", BASE_DIR / "weights", BASE_DIR / "runs"]
    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob("best.pt"))
        if matches:
            return matches[0]
    return None


def install_requirements() -> bool:
    requirements_file = BASE_DIR / "requirements.txt"
    if not requirements_file.exists():
        return False

    with st.spinner("Installing dependencies. This can take a few minutes..."):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        st.error(f"Dependency installation failed.\n{result.stderr or result.stdout}")
        return False
    return True


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    global YOLO

    if YOLO is None:
        st.warning(f"Ultralytics import failed: {IMPORT_ERROR}")
        if install_requirements():
            try:
                from ultralytics import YOLO as reloaded_yolo
            except Exception as retry_error:  # pragma: no cover - handled at runtime
                st.error(f"Ultralytics still could not be imported: {retry_error}")
                return None
            YOLO = reloaded_yolo
        else:
            return None

    model_path = find_model_path()
    if model_path is None:
        st.error("No best.pt model file was found. Please upload it to the repo root or a subfolder.")
        return None

    st.info(f"Using model: {model_path}")
    return YOLO(str(model_path))


def count_detections(model: Any, image_path: str, conf: float) -> int:
    if model is None:
        raise RuntimeError("Model is not available")

    results = model(image_path, conf=conf, stream=False, imgsz=640)
    result = results[0]
    return int(len(result.boxes.cls))


def main() -> None:
    st.title("Florodye Cell Detection")
    st.write("Upload up to 36 images and get the total number of detected cells using your trained YOLO model.")

    model = load_model()
    if model is None:
        st.stop()

    uploaded_files = st.file_uploader(
        "Choose up to 36 images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files is None:
        st.stop()

    if len(uploaded_files) > 36:
        st.error("Please upload 36 images or fewer.")
        st.stop()

    if len(uploaded_files) == 0:
        st.info("No images selected yet.")
        st.stop()

    conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.95, 0.25, 0.05)

    if st.button("Run detection"):
        with st.spinner("Processing images..."):
            total_cells = 0
            per_image_results = []
            temp_files = []

            try:
                for uploaded_file in uploaded_files:
                    suffix = Path(uploaded_file.name).suffix or ".png"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = Path(tmp_file.name)
                    temp_files.append(temp_path)

                    count = count_detections(model, str(temp_path), conf=conf_threshold)
                    total_cells += count
                    per_image_results.append((uploaded_file.name, count))

                st.success(f"Total cells detected across {len(uploaded_files)} images: {total_cells}")
                st.subheader("Per-image results")
                for name, count in per_image_results:
                    st.write(f"- {name}: {count} cells")
            finally:
                for temp_file in temp_files:
                    try:
                        temp_file.unlink(missing_ok=True)
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
