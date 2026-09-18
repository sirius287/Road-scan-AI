import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🚁",
    layout="wide"
)

st.title("🚁 RoadGuard AI")
st.subheader("AI-Powered Drone Road Damage Detection")

# Load models
@st.cache_resource
def load_models():
    pothole_model = YOLO("pothole_best.pt")
    crack_model = YOLO("best.pt")
    return pothole_model, crack_model


pothole_model, crack_model = load_models()

uploaded = st.file_uploader(
    "Upload a road image captured by the ESP32-CAM",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded)

    st.write("### 📷 Captured Road Image")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="ESP32-CAM Image", use_container_width=True)

    # Save temporary image
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:

        image.save(tmp.name)
        image_path = tmp.name

    # -------------------------
    # POTHOLE DETECTION
    # -------------------------

    pothole_results = pothole_model.predict(
        source=image_path,
        device=0,
        conf=0.20,
        imgsz=640,
        verbose=False
    )

    pothole_result = pothole_results[0]

    pothole_count = len(pothole_result.boxes)

    # -------------------------
    # CRACK DETECTION
    # -------------------------

    crack_results = crack_model.predict(
        source=image_path,
        device=0,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    crack_result = crack_results[0]

    crack_count = 0

    for box in crack_result.boxes:
        class_id = int(box.cls[0])

        # 0,1,2 are crack categories
        if class_id in [0, 1, 2]:
            crack_count += 1

    # -------------------------
    # DISPLAY
    # -------------------------

    with col2:

        st.write("### 🤖 AI Analysis")

        if pothole_count > 0:
            st.success(f"🕳️ Pothole detections: {pothole_count}")
        else:
            st.info("No pothole detected")

        if crack_count > 0:
            st.warning(f"⚠️ Crack detections: {crack_count}")
        else:
            st.info("No cracks detected")

        total_damage = pothole_count + crack_count

        if total_damage >= 3:
            severity = "HIGH"
        elif total_damage >= 1:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        st.metric(
            "Damage Severity",
            severity
        )

    # -------------------------
    # ANNOTATED POTHOLE IMAGE
    # -------------------------

    annotated = pothole_result.plot()

    st.write("### 🔍 Detection Result")

    st.image(
        annotated,
        caption="AI-detected road damage",
        use_container_width=True
    )

    # -------------------------
    # REPORT
    # -------------------------

    st.write("### 📋 Damage Report")

    st.write(f"""
**Road Damage Inspection Report**

- Potholes detected: **{pothole_count}**
- Crack detections: **{crack_count}**
- Overall severity: **{severity}**
- Detection device: **ESP32-CAM**
- AI inference device: **Laptop GPU**
- Detection model: **YOLO**
""")

    st.success("Inspection complete.")