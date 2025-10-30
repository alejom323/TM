import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Configuración de página Streamlit
st.set_page_config(
    page_title="Reconocimiento de Imágenes",
    page_icon="🧠",
    layout="centered"
)

# Estilos personalizados: tipografía cursiva Dancing Script
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Dancing+Script&display=swap');

        html, body, [class*="css"] {
            font-family: 'Dancing Script', cursive;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            font-family: 'Dancing Script', cursive !important;
        }

        section[data-testid="stSidebar"] * {
            font-family: 'Dancing Script', cursive !important;
        }
    </style>
""", unsafe_allow_html=True)

# Muestra la versión de Python
st.write("Versión de Python:", platform.python_version())

# Carga del modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título principal
st.title("Reconocimiento de Imágenes")

# Imagen personalizada
image = Image.open('foto_manorobt.jpg')
st.image(image, width=350)

# Barra lateral
with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usar esta app para identificar")

# Captura de imagen desde cámara
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    print(prediction)

    if prediction[0][0] > 0.5:
        st.header('Izquierda, con Probabilidad: ' + str(prediction[0][0]))
    if prediction[0][1] > 0.5:
        st.header('Arriba, con Probabilidad: ' + str(prediction[0][1]))
    #if prediction[0][2] > 0.5:
    #    st.header('Derecha, con Probabilidad: ' + str(prediction[0][2]))

