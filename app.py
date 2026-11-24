
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from PIL import Image


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AgriVision | Tomato Disease Detection",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "custom_cnn.keras"
)

CLASS_MAPPING_PATH = os.path.join(
    BASE_DIR, "class_mapping.json"
)

IMG_SIZE = 224


# ==========================================
# 2. DISEASE INFORMATION
# ==========================================

DISEASE_INFO = {

    "Tomato_Bacterial_spot": {
        "name": "Bacterial Spot",
        "description": (
            "A bacterial disease that causes small, dark, "
            "water-soaked spots on tomato leaves. "
            "Infected areas may develop yellow halos."
        ),
        "recommendations": [
            "Remove heavily affected plant material where appropriate.",
            "Avoid overhead irrigation.",
            "Use clean gardening tools.",
            "Consult local agricultural experts."
        ]
    },

    "Tomato_Early_blight": {
        "name": "Early Blight",
        "description": (
            "A fungal disease commonly associated with brown "
            "lesions that may develop concentric rings, "
            "often appearing on older leaves."
        ),
        "recommendations": [
            "Remove infected lower leaves where appropriate.",
            "Avoid overhead watering.",
            "Maintain adequate plant spacing.",
            "Follow locally recommended disease-management practices."
        ]
    },

    "Tomato_Late_blight": {
        "name": "Late Blight",
        "description": (
            "A serious disease caused by Phytophthora infestans. "
            "It can produce rapidly expanding dark lesions "
            "on leaves under favourable conditions."
        ),
        "recommendations": [
            "Inspect nearby plants for similar symptoms.",
            "Avoid prolonged leaf wetness.",
            "Remove infected plant material appropriately.",
            "Seek agricultural specialist advice."
        ]
    },

    "Tomato_Leaf_Mold": {
        "name": "Leaf Mold",
        "description": (
            "A fungal disease that commonly produces yellowish "
            "areas on the upper leaf surface and olive-green "
            "or brown growth underneath."
        ),
        "recommendations": [
            "Improve ventilation around plants.",
            "Reduce excessive humidity.",
            "Remove severely affected leaves where appropriate.",
            "Follow recommended disease-management practices."
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "description": (
            "A fungal disease that causes numerous small "
            "circular spots, often with pale centres "
            "and darker margins."
        ),
        "recommendations": [
            "Remove affected lower leaves where appropriate.",
            "Avoid splashing water onto foliage.",
            "Maintain good plant spacing.",
            "Monitor disease progression."
        ]
    },

    "Tomato_healthy": {
        "name": "Healthy Tomato Leaf",
        "description": (
            "The model identifies this image as belonging "
            "to the healthy tomato leaf category."
        ),
        "recommendations": [
            "Continue regular plant monitoring.",
            "Maintain appropriate watering practices.",
            "Ensure adequate sunlight and air circulation.",
            "Inspect leaves regularly."
        ]
    }
}


# ==========================================
# 3. LOAD MODEL AND CLASS MAPPING
# ==========================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    with open(CLASS_MAPPING_PATH, "r") as file:
        mapping = json.load(file)

    if isinstance(mapping, dict):

        class_names = sorted(
            mapping.keys(),
            key=lambda name: mapping[name]
        )

    else:

        class_names = mapping

    return model, class_names


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    input_array = np.expand_dims(
        image_array,
        axis=0
    )

    return input_array


# ==========================================
# 5. GRAD-CAM GENERATION
# ==========================================

def generate_gradcam(model, image_array):

    target_layer = model.get_layer("activation_3")

    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=target_layer.output
    )

    target_index = model.layers.index(
        target_layer
    )

    classifier_input = tf.keras.Input(
        shape=target_layer.output.shape[1:]
    )

    x = classifier_input

    for layer in model.layers[target_index + 1:]:

        x = layer(x)

    classifier = tf.keras.Model(
        inputs=classifier_input,
        outputs=x
    )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs = feature_extractor(
            image_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        predictions = classifier(
            conv_outputs,
            training=False
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[:, predicted_class]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Unable to calculate Grad-CAM gradients."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        conv_outputs[0] * weights[0],
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


# ==========================================
# 6. GRAD-CAM VISUALIZATION
# ==========================================

def create_gradcam_visualization(
    original_image,
    heatmap
):

    original = np.array(
        original_image.resize((224, 224))
    )

    heatmap_resized = np.array(
        Image.fromarray(
            np.uint8(heatmap * 255)
        ).resize((224, 224))
    )

    colormap = plt.get_cmap("jet")

    heatmap_color = (
        colormap(heatmap_resized / 255.0)[:, :, :3] * 255
    ).astype(np.uint8)

    overlay = (
        0.6 * original.astype(np.float32)
        + 0.4 * heatmap_color.astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return original, heatmap_color, overlay


# ==========================================
# 7. APPLICATION HEADER
# ==========================================

st.title("🌱 AgriVision")

st.subheader(
    "AI-Powered Tomato Leaf Disease Detection System"
)

st.markdown(
    """
    Upload an image of a tomato leaf to identify its
    disease category using deep learning.

    AgriVision combines image classification with
    Explainable AI to make predictions more interpretable.
    """
)

st.divider()


# ==========================================
# 8. SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown(
    """
    ### Deep Learning Model

    AgriVision uses a trained CNN architecture
    to identify tomato leaf diseases.
    """
)

st.sidebar.divider()

st.sidebar.markdown("### Supported Categories")

for category in DISEASE_INFO:

    st.sidebar.markdown(
        f"- {DISEASE_INFO[category]['name']}"
    )

st.sidebar.divider()

st.sidebar.info(
    "Upload a clear image of a tomato leaf "
    "for disease classification."
)

st.sidebar.caption(
    "AgriVision | Academic Deep Learning Project"
)


# ==========================================
# 9. LOAD MODEL
# ==========================================

try:

    model, CLASS_NAMES = load_resources()

except Exception as error:

    st.error(
        f"Unable to load model resources: {error}"
    )

    st.info(
        "Check the model file and class_mapping.json."
    )

    st.stop()


# ==========================================
# 10. IMAGE UPLOAD
# ==========================================

st.header("📸 Upload Tomato Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a tomato leaf photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        original_image,
        caption="Input Tomato Leaf",
        width=400
    )

    st.divider()

    if st.button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analysing tomato leaf image..."
        ):

            try:

                # ==================================
                # 11. IMAGE PREPROCESSING
                # ==================================

                input_array = preprocess_image(
                    original_image
                )

                # ==================================
                # 12. DISEASE PREDICTION
                # ==================================

                predictions = model.predict(
                    input_array,
                    verbose=0
                )[0]

                if len(predictions) != len(CLASS_NAMES):

                    raise ValueError(
                        "Model output dimensions do not "
                        "match the class mapping."
                    )

                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                confidence = float(
                    predictions[predicted_index] * 100
                )

                # ==================================
                # 13. PREDICTION RESULT
                # ==================================

                st.header("📊 Prediction Results")

                st.success(
                    f"Predicted Disease: "
                    f"{DISEASE_INFO[predicted_class]['name']}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.progress(
                        min(max(confidence / 100, 0), 1)
                    )

                with col2:

                    st.metric(
                        "Predicted Category",
                        predicted_index + 1
                    )

                st.divider()

                # ==================================
                # 14. CLASS-WISE PROBABILITY
                # ==================================

                st.header(
                    "📈 Class-wise Probability Distribution"
                )

                display_names = [
                    DISEASE_INFO[name]["name"]
                    for name in CLASS_NAMES
                ]

                probabilities = predictions * 100

                fig, ax = plt.subplots(
                    figsize=(11, 5)
                )

                bars = ax.bar(
                    display_names,
                    probabilities
                )

                ax.set_ylabel(
                    "Probability (%)"
                )

                ax.set_xlabel(
                    "Disease Categories"
                )

                ax.set_title(
                    "Disease Classification Probability"
                )

                ax.set_ylim(0, 105)

                ax.set_xticks(
                    range(len(display_names))
                )

                ax.set_xticklabels(
                    display_names,
                    rotation=25,
                    ha="right"
                )

                ax.bar_label(
                    bars,
                    fmt="%.2f%%",
                    padding=3
                )

                ax.grid(
                    axis="y",
                    alpha=0.3
                )

                ax.set_axisbelow(True)

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

                # ==================================
                # 15. DETAILED PROBABILITY TABLE
                # ==================================

                st.subheader(
                    "Detailed Probability Scores"
                )

                probability_df = pd.DataFrame({

                    "Disease Category": display_names,

                    "Probability (%)": probabilities

                })

                probability_df = probability_df.sort_values(
                    by="Probability (%)",
                    ascending=False
                )

                st.dataframe(
                    probability_df.style.format({
                        "Probability (%)": "{:.2f}%"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                # ==================================
                # 16. DISEASE INFORMATION
                # ==================================

                st.divider()

                st.header("🌿 Disease Information")

                disease_info = DISEASE_INFO[
                    predicted_class
                ]

                st.subheader(
                    disease_info["name"]
                )

                st.write(
                    disease_info["description"]
                )

                st.markdown(
                    "### General Management Recommendations"
                )

                for recommendation in disease_info[
                    "recommendations"
                ]:

                    st.markdown(
                        f"- {recommendation}"
                    )

                # ==================================
                # 17. GRAD-CAM EXPLAINABILITY
                # ==================================

                st.divider()

                st.header(
                    "🔬 Explainable AI - Grad-CAM"
                )

                heatmap = generate_gradcam(
                    model,
                    input_array
                )

                original, heatmap_color, overlay = (
                    create_gradcam_visualization(
                        original_image,
                        heatmap
                    )
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.image(
                        original,
                        caption="Original Leaf",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        heatmap_color,
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )

                with col3:

                    st.image(
                        overlay,
                        caption="Grad-CAM Overlay",
                        use_container_width=True
                    )

                st.caption(
                    "Grad-CAM highlights image regions "
                    "that influence the prediction. "
                    "These regions are not confirmed "
                    "disease boundaries."
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

                st.exception(error)


# ==========================================
# 18. FOOTER
# ==========================================

st.divider()

st.caption(
    """
    AgriVision is an academic image-classification
    prototype trained on six tomato leaf categories
    from the PlantVillage dataset.

    Predictions are experimental and should not be
    treated as definitive agricultural diagnoses.
    Real-world images may produce different results.
    Consult an agricultural specialist before making
    crop-management decisions.
    """
)



