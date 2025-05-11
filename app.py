from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("Pneumonia_model.h5")

CLASSES = ['Normal', 'Pneumonia']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index])
    return CLASSES[class_index], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            filepath = os.path.join("static", filename)
            file.save(filepath)
            label, confidence = predict_image(filepath)
            return render_template("index.html", label=label, confidence=confidence, image=filename)
    return render_template("index.html", label=None)

if __name__ == "__main__":
    app.run(debug=True)
