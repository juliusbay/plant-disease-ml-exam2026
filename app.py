import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

MODEL_PATH = "plant_disease_model.h5"

model = None
class_names = None


def get_model():
    global model, class_names
    if model is None:
        model = load_model(MODEL_PATH)
        # Derive class count from the output layer and build sorted placeholder names,
        # but the real class list is injected via CLASS_NAMES env var or the 38-class
        # standard New Plant Diseases Dataset ordering defined below.
        num_classes = model.output_shape[-1]
        class_names = _build_class_names(num_classes)
    return model, class_names


# Standard 38-class ordering from the New Plant Diseases Dataset (Kaggle).
# The ImageDataGenerator sorts directory names alphabetically, so this list
# mirrors that exact sort order.
_STANDARD_38 = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def _build_class_names(num_classes: int) -> list[str]:
    env_names = os.environ.get("CLASS_NAMES", "")
    if env_names:
        names = [n.strip() for n in env_names.split(",") if n.strip()]
        if len(names) == num_classes:
            return names

    if num_classes == 38:
        return _STANDARD_38

    # Fallback: generic labels
    return [f"Class_{i}" for i in range(num_classes)]


def _format_label(raw: str) -> tuple[str, str]:
    """Return (plant, condition) from a raw class name like 'Tomato___Early_blight'."""
    parts = raw.split("___")
    plant = parts[0].replace("_", " ").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return plant, condition


def predict(image_path: str) -> dict:
    mdl, names = get_model()
    img = load_img(image_path, target_size=(224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = mdl.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100
    raw = names[idx]
    plant, condition = _format_label(raw)
    return {
        "raw_class": raw,
        "plant": plant,
        "condition": condition,
        "confidence": round(confidence, 2),
        "is_healthy": "healthy" in raw.lower(),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="Please select an image file.")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            return render_template("index.html", error="Unsupported file type. Use JPG, PNG, or WebP.")

        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        result = predict(save_path)
        result["image_url"] = url_for("static", filename=f"../uploads/{filename}")
        # Serve the upload via a dedicated route instead
        result["image_filename"] = filename
        return render_template("index.html", result=result)

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
