import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

def title():
    st.title("🔥Welcome to the Deep Learning-Based Web Application🔥")
    st.markdown("Deep learning is a subset of machine learning that uses artificial neural networks to process and analyze complex data. It enables computers to learn patterns and make decisions with minimal human intervention.")
    st.image("images/title_image.png",use_container_width=True)

# Sidebar
def sidebar_setup():
    selected_model = None

    with st.sidebar:
        st.markdown("# 🌾🌾Deep Rice")
        st.success("Welcome")
        st.markdown("#### This web application accurately classifies rice varieties and detects leaf diseases. Simply select a dataset and upload an image for analysis.")

        st.subheader("Models and Datasets")
        model = st.selectbox("Select a model",(
            "Rice Varieties (Model_1)",
            "Rice Varieties (Model_2)",
            "Leaf Diseases (Model_3)",
            "Leaf Diseases (Model_4)"),
            index=0)
        
        # Custom CSS for the button
        st.markdown("""
            <style>
            .stButton>button {
                width: 50%;
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 5px;
                border: none;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            </style>
            """, unsafe_allow_html=True)
        
        if st.button("Submit",type="primary"):
            if model == "Rice Varieties (Model_1)":
                selected_model = 1
            elif model == "Rice Varieties (Model_2)":
                selected_model = 2
            elif model == "Leaf Diseases (Model_3)":
                selected_model = 3
            elif model == "Leaf Diseases (Model_4)":
                selected_model = 4
            else:
                st.write("!!! Wrong Input")

            st.session_state["submitted"] = True
            st.session_state["selected_model"] = selected_model

        return selected_model
    
@st.cache_resource
def load_model(model_location:str):
    model = tf.keras.models.load_model(model_location)
    return model

def preprocess_image(image):
    img = image.resize((224,224))
    # img_array = np.array(img)
    # img_array = img_array/255.0
    img_array = tf.expand_dims(img, axis = 0)
    return img_array

def predict_class(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    prediction_class = np.argmax(prediction)
    return prediction_class


    
def upload_image(class_names, model):
    uploaded_img = st.file_uploader("Choose an image to classify", type=["jpg","png","jpeg"], accept_multiple_files=False)

    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        img = img.resize((200,200))
        st.image(img, caption = "Uploaded Image")
        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Classify", type="secondary"):
                with col2:
                    result_placeholder = st.empty()  
                    with st.spinner("Classifying..."): 
                        class_id = predict_class(img, model)
                        result_placeholder.success(f"###### Predicted Class: {class_names[class_id]}")


# Model 1
def model_1():
    model_location = "models\model_dataset_1.keras" # u can change
    model = load_model(model_location)

    st.title("Welcome to the Rice Variety Classifier for Dataset 1")
    st.markdown("The dataset includes 20 distinct rice varieties: Subol Lota, Bashmoti (Deshi), Ganjiya, Shampakatari, Sugandhi Katarivog, BR-28, BR-29, Paijam, Bashful, Lal Aush, BR-Jirashail, Gutisharna, Birui, Najirshail, Pahari Birui, Polao (Katari), Polao (Chinigura), Amon, Shorna-5, and Lal Binni. You can classify any rice variety by uploading an image.")

    with st.expander("Check Trained Model Performance"):
        train_accuracy = "100%"
        test_accuracy = "99.89%"

        st.write(f"Train Accuracy : {train_accuracy}")
        st.write(f"Test Accuracy : {test_accuracy}")

        st.write("Loss Curve: ")
        loss_img_loc = "images/curve_1.png"
        img = Image.open(loss_img_loc)
        st.image(img, use_container_width=True)

        st.write("Confusion Matrix : ")
        conf_img_loc = "images\confusion_matrix_1.png"
        img = Image.open(conf_img_loc)
        st.image(img, use_container_width=True)

    # Real job
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
    
    upload_image(class_names, model)


# continue from here
def model_2():
    model_location = "models\model_dataset_2.keras" # u can change
    model = load_model(model_location)

    st.title("Welcome to the Rice Variety Classifier for Dataset 2")
    st.markdown("The dataset includes five distinct rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. You can classify any rice variety by uploading an image.")

    with st.expander("Check Trained Model Performance"):
        train_accuracy = "100%"
        test_accuracy = "99.82%"

        st.write(f"Train Accuracy : {train_accuracy}")
        st.write(f"Test Accuracy : {test_accuracy}")

        st.write("Loss Curve: ")
        loss_img_loc = "images/curve_2.png"
        img = Image.open(loss_img_loc)
        st.image(img, use_container_width=True)

        st.write("Confusion Matrix : ")
        conf_img_loc = "images\confusion_matrix_2.png"
        img = Image.open(conf_img_loc)
        st.image(img, use_container_width=True)

    # Real job
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    
    upload_image(class_names, model)

def model_3():
    pass

def model_4():
    pass



if __name__ == "__main__":

    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

    sidebar_setup()

    main_placeholder = st.empty()
    
    if st.session_state["submitted"] is False:
        with main_placeholder.container():
            title()
    else:
        with main_placeholder.container():
            model = st.session_state["selected_model"]
            # Model configaration
            if model == 1:
                model_1()
            elif model == 2:
                model_2()
            elif model == 3:
                model_3()
            elif model == 4:
                model_4()
            else:
                pass

    
