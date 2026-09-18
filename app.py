import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
from pathlib import Path

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🚁",
    layout="wide"
)

st.title("🚁 RoadGuard AI")
st.subheader("AI-Powered Drone Road Damage Detection")

# -------------------------
# LOAD MODELS
# -------------------------

@st.cache_resource
def load_models():

    base_dir = Path(__file__).parent

    # Pothole model
    pothole_model = YOLO(str(base_dir / "pothole_best.pt"))

    # Crack model is optional
    crack_path = base_dir / "best.pt"

    if crack_path.exists():
        crack_model = YOLO(str(crack_path))
    else:
        crack_model = None

    return pothole_model, crack_model


pothole_model, crack_model = load_models()

# -------------------------
# UPLOAD IMAGE
# -------------------------

uploaded = st.file_uploader(
    "Upload a road image captured by the ESP32-CAM",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.write("### 📷 Captured Road Image")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="ESP32-CAM Image",
            use_container_width=True
        )

    # -------------------------
    # SAVE TEMPORARY IMAGE
    # -------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:

        image.save(tmp.name)
        image_path = tmp.name

    # -------------------------
    # POTHOLE DETECTION
    # -------------------------

    with st.spinner("🤖 Running AI detection..."):

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

    crack_count = 0

    if crack_model is not None:

        crack_results = crack_model.predict(
            source=image_path,
            device=0,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        crack_result = crack_results[0]

        for box in crack_result.boxes:

            class_id = int(box.cls[0])

            # 0, 1, 2 = crack categories
            if class_id in [0, 1, 2]:
                crack_count += 1

    # -------------------------
    # SEVERITY
    # -------------------------

    total_damage = pothole_count + crack_count

    if total_damage >= 3:
        severity = "HIGH"
    elif total_damage >= 1:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # -------------------------
    # DISPLAY ANALYSIS
    # -------------------------

    with col2:

        st.write("### 🤖 AI Analysis")

        if pothole_count > 0:
            st.success(
                f"🕳️ Pothole detections: {pothole_count}"
            )
        else:
            st.info("No pothole detected")

        if crack_model is not None:

            if crack_count > 0:
                st.warning(
                    f"⚠️ Crack detections: {crack_count}"
                )
            else:
                st.info("No cracks detected")

        else:

            st.info(
                "ℹ️ Crack detection model not installed"
            )

        st.metric(
            "Damage Severity",
            severity
        )

    # -------------------------
    # ANNOTATED IMAGE
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
- AI inference device: **NVIDIA GPU**
- Detection model: **YOLO**
""")

    if crack_model is None:
        st.caption(
            "Crack detection is currently disabled because best.pt "
            "is not present in the project folder."
        )

    st.success("✅ Inspection complete.")