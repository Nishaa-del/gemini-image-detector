import os
import numpy as np
from PIL import Image
import streamlit as st
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

# -----------------------------
# STEP 1: Create synthetic data
# -----------------------------
def create_synthetic_images(n_samples=50, img_size=(64, 64)):
    os.makedirs("dataset/real", exist_ok=True)
    os.makedirs("dataset/gemini", exist_ok=True)

    for i in range(n_samples):
        # Real images (random grayscale)
        real_img = np.random.randint(100, 200, (img_size[0], img_size[1], 3), dtype=np.uint8)
        Image.fromarray(real_img).save(f"dataset/real/real_{i}.jpg")

        # Gemini images (patterned synthetic)
        gemini_img = np.random.randint(0, 100, (img_size[0], img_size[1], 3), dtype=np.uint8)
        gemini_img[:, :, 0] = np.clip(gemini_img[:, :, 0] + 80, 0, 255)  # Add red tint
        Image.fromarray(gemini_img).save(f"dataset/gemini/gemini_{i}.jpg")

# -----------------------------
# STEP 2: Load dataset
# -----------------------------
def load_dataset():
    X, y = [], []
    classes = ['real', 'gemini']
    for label, class_name in enumerate(classes):
        folder = os.path.join('dataset', class_name)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize((32, 32))
            X.append(np.array(img).flatten())
            y.append(label)
    return np.array(X), np.array(y), classes

# -----------------------------
# STEP 3: Train SVM Model
# -----------------------------
def train_model():
    create_synthetic_images()
    X, y, classes = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump({'model': model, 'classes': classes}, 'svm_model.pkl')
    return acc

# -----------------------------
# STEP 4: Prediction Function
# -----------------------------
def predict_image(image):
    data = joblib.load('svm_model.pkl')
    model = data['model']
    classes = data['classes']
    img = image.resize((32, 32))
    features = np.array(img).flatten().reshape(1, -1)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    return classes[pred], prob

# -----------------------------
# STEP 5: Streamlit Web App
# -----------------------------
st.title("🧠 AI Image Detector")
st.write("Upload an image to detect if it’s **AI-generated** or **Real**.")

# Train model (only once)
if not os.path.exists("svm_model.pkl"):
    with st.spinner("Training model, please wait..."):
        acc = train_model()
        st.success(f"✅ Model trained successfully (Accuracy: {acc*100:.2f}%)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    label, prob = predict_image(image)
    st.write(f"### 🔍 Predicted Class: **{label.upper()}**")
    st.write(f"Confidence: {prob}")

    if label == "gemini":
        st.error("⚠️ This image is likely gemini-generated!")
    else:
        st.success("✅ This image appears to be real.")
