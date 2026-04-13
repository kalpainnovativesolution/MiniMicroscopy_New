import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import gdown

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(layout="wide")

# -------------------------------
# Google Drive Model Config
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1Ffo8RgXxe2Tm4FZIIm1_TTUOK5AvLZdd"
MODEL_PATH = "MiniMicroscope_Model_Updated.pt"

# -------------------------------
# Download model from Google Drive (only once)
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------------
# Detection Function
# -------------------------------
def detect_and_annotate(image, conf_thresh, show_boxes, show_labels, show_conf, overlap_thresh, opacity_thresh):
    results = model(image, conf=conf_thresh, iou=overlap_thresh)[0]

    total_count = len(results.boxes)
    sc_count = 0
    usc_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name == "SC":
            sc_count += 1
        elif class_name == "USC":
            usc_count += 1

    annotated_img = image.copy()

    if show_boxes or show_labels or show_conf:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf_score = float(box.conf[0])

            if show_boxes:
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = ""
            if show_labels:
                label_text += f"{class_name}"
            if show_conf:
                if label_text:
                    label_text += f" {conf_score:.2f}"
                else:
                    label_text += f"{conf_score:.2f}"

            if label_text:
                overlay = annotated_img.copy()
                cv2.putText(
                    overlay,
                    label_text,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1
                )
                annotated_img = cv2.addWeighted(
                    overlay,
                    opacity_thresh,
                    annotated_img,
                    1 - opacity_thresh,
                    0
                )

    return total_count, sc_count, usc_count, annotated_img


# -------------------------------
# UI
# -------------------------------
st.title("🔬 Mini-Microscopy Somatic Cell Detection")
st.write("Upload an image to count stained and unstained cells.")

# Sidebar
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
overlap_thresh = st.sidebar.slider("Overlap (IoU) Threshold", 0.1, 1.0, 0.45, 0.05)
opacity_thresh = st.sidebar.slider("Annotation Opacity", 0.1, 1.0, 0.5, 0.05)

show_boxes = st.sidebar.checkbox("Display Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Display Cell Class Labels", value=True)
show_conf = st.sidebar.checkbox("Display Confidence Score", value=False)

# Upload
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"]
)

if uploaded_file is not None:
    # Read original image
    pil_image = Image.open(uploaded_file).convert("RGB")
    pil_image = pil_image.resize((640, 640))

    # Original image for display
    original_rgb = np.array(pil_image)

    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

    # -------------------------------
    # Grayscale preprocessing
    # -------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Convert grayscale back to 3-channel image for YOLO inference
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Detection on grayscale-preprocessed image
    total_count, sc_count, usc_count, output_img = detect_and_annotate(
        gray_3ch,
        conf_thresh,
        show_boxes,
        show_labels,
        show_conf,
        overlap_thresh,
        opacity_thresh
    )

    # KPI cards
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div style='background-color:#FFDDC1;padding:10px;border-radius:10px;text-align:center;'><h3>Total Cells</h3><h2>{total_count}</h2></div>",
        unsafe_allow_html=True
    )
    col2.markdown(
        f"<div style='background-color:#C1FFD7;padding:10px;border-radius:10px;text-align:center;'><h3>Stained Cells (SC)</h3><h2>{sc_count}</h2></div>",
        unsafe_allow_html=True
    )
    col3.markdown(
        f"<div style='background-color:#C1D4FF;padding:10px;border-radius:10px;text-align:center;'><h3>Unstained Cells (USC)</h3><h2>{usc_count}</h2></div>",
        unsafe_allow_html=True
    )

    # Convert predicted image for display
    output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    st.subheader("Visualization")

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_rgb, caption="Original Image (Resized 640x640)", use_container_width=True)
    with col2:
        st.image(output_img_rgb, caption="Predicted Image with Annotations", use_container_width=True)
