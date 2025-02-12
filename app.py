import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os


st.set_page_config(menu_items={})

# MODEL_PATHS = {
#     "model_1": "models/model_dataset_1.keras",
#     "model_2": "models/model_dataset_2.keras",
#     "model_3": "models/model_dataset_3.keras",
#     "model_4": "models/model_dataset_4.keras",
# }

MODEL_PATHS = {
    "model_1": os.path.join("models","model_dataset_1.keras"),
    "model_2": os.path.join("models","model_dataset_2.keras"),
    "model_3": os.path.join("models","model_dataset_3.keras"),
    "model_4": os.path.join("models","model_dataset_4.keras"),
}

def title():
    st.title("🔥Welcome to the Deep Learning-Based Web Application🔥")
    st.markdown("Deep learning is a subset of machine learning that uses artificial neural networks to process and analyze complex data. It enables computers to learn patterns and make decisions with minimal human intervention.")
    st.image(os.path.join("images","title_image.png"),use_container_width=True)

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
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model File not found at {model_path}. Please check the path.")
        return None
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    img = image.resize((224,224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis = 0)
    return img_array

def predict_class(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    prediction_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return prediction_class, confidence

def choose_method(class_names, model):
    st.markdown("#### Choose an Option")
    option = st.radio("Select an option", ("Use Camera", "Upload an image"))

    if st.button("Select"):
        st.session_state["option"] = option

    if "option" in st.session_state:
        if st.session_state["option"] == "Use Camera":
            st.markdown("#### Take a picture")
            picture = st.camera_input("Take a picture")

            if picture is not None:
                img = Image.open(picture)
                img = img.resize((224,224))
                st.image(img, caption = "Uploaded Image")
                classify_image(img, class_names, model)
            
        elif st.session_state["option"] == "Upload an image":
            st.markdown("Upload an image")
            uploaded_img = st.file_uploader("Choose an image to classify", type=["jpg","png","jpeg"], accept_multiple_files=False)

            if uploaded_img is not None:
                img = Image.open(uploaded_img)
                img = img.resize((224,224))
                st.image(img, caption = "Uploaded Image")
                classify_image(img, class_names, model)


def classify_image(img,class_names, model):

    # if uploaded_img is not None:
    #     img = Image.open(uploaded_img)
    #     img = img.resize((224,224))
    #     st.image(img, caption = "Uploaded Image")
        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Classify", type="secondary"):
                with col2:
                    result_placeholder = st.empty()  
                    with st.spinner("Classifying..."): 
                        class_id, confidance = predict_class(img, model)
                        if confidance < 0.5:
                            result_placeholder.warning("###### Warning: The confidence level is below 50%. Consider using a clearer image for better accuracy or the image may not belong to our model classes.")
                        else:
                            confidance = confidance * 100
                            result_placeholder.success(f"###### Predicted Class: {class_names[class_id]} \n ###### Confidence Level: {confidance:.2f}%")


    
# def upload_image(class_names, model):
#     # uploaded_img = st.file_uploader("Choose an image to classify", type=["jpg","png","jpeg"], accept_multiple_files=False)
#     uploaded_img = choose_method()

#     if uploaded_img is not None:
#         img = Image.open(uploaded_img)
#         img = img.resize((224,224))
#         st.image(img, caption = "Uploaded Image")
        
#         col1, col2 = st.columns(2)

#         with col1:
#             if st.button("Classify", type="secondary"):
#                 with col2:
#                     result_placeholder = st.empty()  
#                     with st.spinner("Classifying..."): 
#                         class_id, confidance = predict_class(img, model)
#                         if confidance < 0.5:
#                             result_placeholder.warning("###### Warning: The confidence level is below 50%. Consider using a clearer image for better accuracy or the image may not belong to our model classes.")
#                         else:
#                             confidance = confidance * 100
#                             result_placeholder.success(f"###### Predicted Class: {class_names[class_id]} \n ###### Confidence Level: {confidance:.2f}%")


# Model 1
def model_1():

    model_location = MODEL_PATHS.get("model_1") # u can change
    if model_location is None:
        st.error("Invalid Model Selected. Please Try Again")
        return

    model = load_model(model_location)
    if model is None:
        st.warning("Model Load Failed")
        return

    st.title("Welcome to the Rice Variety Classifier for Dataset 1")
    st.markdown("The dataset includes 20 distinct rice varieties: Subol Lota, Bashmoti (Deshi), Ganjiya, Shampakatari, Sugandhi Katarivog, BR-28, BR-29, Paijam, Bashful, Lal Aush, BR-Jirashail, Gutisharna, Birui, Najirshail, Pahari Birui, Polao (Katari), Polao (Chinigura), Amon, Shorna-5, and Lal Binni. You can classify any rice variety by uploading an image.")

    with st.expander("Check Trained Model Performance"):
        train_accuracy = "100%"
        test_accuracy = "99.89%"

        st.write(f"Train Accuracy : {train_accuracy}")
        st.write(f"Test Accuracy : {test_accuracy}")

        st.write("Loss Curve: ")
        loss_img_loc = os.path.join("images","curve_1.png")
        img = Image.open(loss_img_loc)
        st.image(img, use_container_width=True)

        st.write("Confusion Matrix : ")
        conf_img_loc = os.path.join("images","confusion_matrix_1.png")
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
    
    # upload_image(class_names, model)
    choose_method(class_names, model)


# continue from here
def model_2():
    model_location = MODEL_PATHS.get("model_2") # u can change
    if model_location is None:
        st.error("Invalid Model Selected. Please Try Again")
        return

    model = load_model(model_location)
    if model is None:
        st.warning("Model Load Failed")
        return

    st.title("Welcome to the Rice Variety Classifier for Dataset 2")
    st.markdown("The dataset includes five distinct rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. You can classify any rice variety by uploading an image.")

    with st.expander("Check Trained Model Performance"):
        train_accuracy = "100%"
        test_accuracy = "99.82%"

        st.write(f"Train Accuracy : {train_accuracy}")
        st.write(f"Test Accuracy : {test_accuracy}")

        st.write("Loss Curve: ")
        loss_img_loc = os.path.join("images","curve_2.png")
        img = Image.open(loss_img_loc)
        st.image(img, use_container_width=True)

        st.write("Confusion Matrix : ")
        conf_img_loc = os.path.join("images","confusion_matrix_2.png")
        img = Image.open(conf_img_loc)
        st.image(img, use_container_width=True)

    # Real job
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    
    # upload_image(class_names, model)
    choose_method(class_names, model)

def model_3():
    pass

def model_4():
    model_location = MODEL_PATHS.get("model_4") # u can change
    if model_location is None:
        st.error("Invalid Model Selected. Please Try Again")
        return

    model = load_model(model_location)
    if model is None:
        st.warning("Model Load Failed")
        return

    st.title("Welcome to the Rice Leaf Disease Classifier for Dataset 4")
    st.markdown("The dataset includes nine distinct rice leaf diseases: Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf Scald, Narrow Brown Leaf Spot, Neck Blast, Rice Hispa, and Sheath Blight. You can classify any rice leaf disease by uploading an image.")

    with st.expander("Check Trained Model Performance"):
        train_accuracy = "99.99%"
        test_accuracy = "98.55%"

        st.write(f"Train Accuracy : {train_accuracy}")
        st.write(f"Test Accuracy : {test_accuracy}")

        st.write("Loss Curve: ")
        loss_img_loc = os.path.join("images","curve_4.png")
        img = Image.open(loss_img_loc)
        st.image(img, use_container_width=True)

        st.write("Confusion Matrix : ")
        conf_img_loc = os.path.join("images","confusion_matrix_4.png")
        img = Image.open(conf_img_loc)
        st.image(img, use_container_width=True)

    # Real job
    class_names = ['Bacterial Leaf Blight',
                    'Brown Spot',
                    'Healthy Rice Leaf',
                    'Leaf Blast',
                    'Leaf scald',
                    'Narrow Brown Leaf Spot',
                    'Neck_Blast',
                    'Rice Hispa',
                    'Sheath Blight']
    
    # upload_image(class_names, model)
    choose_method(class_names, model)



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

    
