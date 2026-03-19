import streamlit as st
import tempfile
import cv2
import numpy as np

from utils.prediction import final_prediction

# ------------------ SESSION STATE ------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "image" not in st.session_state:
    st.session_state.image = None

# ------------------ UI ------------------
st.title("Traffic Signal Car Colour Detection")
st.write("Upload an image to detect cars, people, and car colors")

# ------------------ SIDEBAR ------------------
st.sidebar.header("Settings")
st.sidebar.write("YOLOv8 + CNN Pipeline")

# ------------------ UPLOAD SECTION ------------------
st.header("Upload Image")

uploaded_file = st.file_uploader(
    "Upload a traffic image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image")

    if st.button("Run Detection", type="primary"):

        with st.spinner("Processing..."):

            try:
                # Save temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                temp_file.write(file_bytes)
                temp_path = temp_file.name

                # Call backend
                output_image, car_count, people_count = final_prediction(temp_path)

                st.session_state.result = {
                    "cars": car_count,
                    "people": people_count
                }

                st.session_state.image = output_image

            except Exception as e:
                st.error(f"Error during processing: {e}")

# ------------------ RESULT DISPLAY ------------------
if st.session_state.result is not None:

    st.subheader("Detection Result")

    col1, col2 = st.columns(2)

    col1.metric("Cars Detected", st.session_state.result["cars"])
    col2.metric("People Detected", st.session_state.result["people"])

    st.image(
        cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB),
        caption="Processed Image",
        use_column_width=True
    )

# ------------------ FOOTER ------------------
st.markdown("*YOLOv8 + CNN | Car Colour Detection System*")