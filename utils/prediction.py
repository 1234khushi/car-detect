import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------- LOAD MODELS ----------------

yolo_model = YOLO(str(BASE_DIR / "models" / "yolov8m.pt"))
cnn_model = tf.keras.models.load_model(str(BASE_DIR / "car_color_cnn_model.h5"))

IMG_SIZE = 64

# -------- LOAD COLOR CLASSES --------
with open(BASE_DIR / "color_classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

color_map = {i: label for i, label in enumerate(classes)}

# ---------------- MAIN FUNCTION ----------------

def final_prediction(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None, 0, 0

    results = yolo_model(image)[0]

    car_count = 0
    people_count = 0

    for box in results.boxes:

        cls = int(box.cls)
        conf = float(box.conf)

        label = yolo_model.names[cls]

        # Only car and person
        if label not in ["car", "person"]:
            continue

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        width = x2 - x1
        height = y2 - y1

        if width < 30 or height < 30:
            continue

        # -------- PEOPLE --------
        if label == "person":
            people_count += 1
            continue

        # -------- CARS --------
        if label == "car":

            car_count += 1

            car_crop = image[y1:y2, x1:x2]

            if car_crop is None or car_crop.size == 0:
                continue

            try:
                car_resized = cv2.resize(car_crop, (IMG_SIZE, IMG_SIZE))
            except:
                continue

            car_normalized = car_resized / 255.0
            car_input = np.expand_dims(car_normalized, axis=0)

            prediction = cnn_model.predict(car_input, verbose=0)
            pred_class = int(np.argmax(prediction))

            color = color_map.get(pred_class, "unknown")

            # -------- COLOR RULE --------
            if "blue" in color.lower():
                box_color = (0,0,255)   # RED
            else:
                box_color = (255,0,0)   # BLUE

            # Draw box
            cv2.rectangle(image, (x1,y1), (x2,y2), box_color, 2)

            cv2.putText(
                image,
                color,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

    # -------- DISPLAY COUNTS --------
    cv2.putText(image, f"Cars: {car_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(image, f"People: {people_count}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return image, car_count, people_count
