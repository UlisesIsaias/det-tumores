from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import os
import gdown  # Asegúrate de tener gdown instalado

app = Flask(__name__)

# Ruta para el modelo en el servidor
MODEL_PATH = "models/classifier-resnet-weights.h5"

# Función para descargar el modelo si no está presente
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado. Descargando...")
        url = 'https://drive.google.com/uc?export=download&id=1Q-pljUVzesRRgliqVP8kQou6J_z4X3Sj'  # URL de tu modelo
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("Modelo ya descargado. Usando modelo existente.")

# Cargar el modelo entrenado
download_model()
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    image = cv2.resize(image, (256, 256))  # Redimensionar a 256x256
    image = np.array(image) / 255.0  # Normalizar
    return np.expand_dims(image, axis=0)  # Añadir dimensión de batch

def encode_image(image):
    """Convierte la imagen en base64 para mostrarla en HTML"""
    img_io = io.BytesIO()
    plt.imsave(img_io, image, format='png')
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if "file" not in request.files:
            return render_template("index.html", error="No se envió ninguna imagen")

        file = request.files["file"]
        image = Image.open(file.stream)
        image = np.array(image)

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        has_tumor = prediction[0][0] > 0.5

        # Generar imágenes
        mask = np.zeros_like(image[:, :, 0])  # Placeholder para la máscara
        if has_tumor:
            mask[image[:, :, 0] > 100] = 255  # Simulación de detección

        image_with_mask = image.copy()
        image_with_mask[mask == 255] = (255, 0, 0)

        # Convertir imágenes a base64 para mostrar en la página
        img_original = encode_image(image)
        img_mask = encode_image(mask)
        img_masked = encode_image(image_with_mask)

        return render_template("index.html", 
                               prediction="Tumor detectado" if has_tumor else "No se detectó tumor",
                               img_original=img_original, 
                               img_mask=img_mask, 
                               img_masked=img_masked)

    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)
