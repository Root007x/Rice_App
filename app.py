import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/model_dataset_1.keras")
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #           optimizer=tf.keras.optimizers.Adam(learning_rate = 0.00002, weight_decay=0.001),
    #           metrics = ['accuracy'])
    return model

model = load_model()

def preprocess_image(image):
    img = image.resize((224,224))
    # img_array = np.array(img)
    # img_array = img_array/255.0
    img_array = tf.expand_dims(img, axis = 0)
    return img_array


def predict_class(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    prediction_class = np.argmax(prediction)
    return prediction_class




class_names = ['10_Lal_Aush',
 '11_Jirashail',
 '12_Gutisharna',
 '13_Red_Cargo',
 '14_Najirshail',
 '15_Katari_Polao',
 '16_Lal_Biroi',
 '17_Chinigura_Polao',
 '18_Amon',
 '19_Shorna5',
 '1_Subol_Lota',
 '20_Lal_Binni',
 '2_Bashmoti',
 '3_Ganjiya',
 '4_Shampakatari',
 '5_Katarivog',
 '6_BR28',
 '7_BR29',
 '8_Paijam',
 '9_Bashful']


st.title("Rice Varieties Image Classification App")

uploaded_img = st.file_uploader("Upload an image", type=["jpg","png","jpeg"], accept_multiple_files=False)

if uploaded_img is not None:
    img = Image.open(uploaded_img)
    st.image(img, caption = "Uploaded Image", use_container_width=True)


    if st.button("Submit"):
        class_id = predict_class(img)
        st.write(f"{class_names[class_id]}")
        st.write(f"{class_id}")

