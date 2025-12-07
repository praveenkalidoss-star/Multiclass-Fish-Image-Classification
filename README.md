# 🐟 Multiclass Fish Image Classification Using Deep Learning

This project builds an AI-based image classification system that identifies different fish species using Deep Learning and Transfer Learning techniques.  
Multiple models were trained and compared, and the best-performing model (MobileNetV2) was deployed using Streamlit for real-time predictions.

---

## 📌 Project Overview

- Preprocess and augment fish images for robust training  
- Train multiple models: CNN (baseline), VGG16, ResNet50, InceptionV3, EfficientNetB0, MobileNetV2  
- Evaluate accuracy, loss, precision, recall, F1-score, and confusion matrix  
- Select the best model based on performance  
- Deploy the final model using a Streamlit web application  
- Provide confidence score + probability distribution for each prediction  

---

## 🎯 Business Use Cases

### ✔ Enhanced Accuracy  
Train multiple models and choose the architecture that gives the highest performance for fish image classification.

### ✔ Deployment Ready  
Develop a user-friendly Streamlit app that allows anyone to upload a fish image and get instant predictions.

### ✔ Model Comparison  
Evaluate all models using validation accuracy, confusion matrix, and F1-score to select the most suitable deep learning model.

---

## 🧠 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **scikit-learn**
- **Matplotlib**
- **Streamlit**
- **Pillow (PIL)**

---

## 🗂️ Dataset

The dataset contains multiple fish species, each stored inside separate folders.

### Folder Structure:

dataset/
train/
val/
test/


Loaded using **ImageDataGenerator** with augmentation (rotation, zoom, flip, rescale).

---

## 🚀 Model Training

We trained the following models:

- CNN (baseline)
- **VGG16**
- **ResNet50**
- **InceptionV3**
- **EfficientNetB0**
- **MobileNetV2** → ✔ BEST MODEL

After detailed comparison, **MobileNetV2** achieved:

- **Training Accuracy:** ~98%  
- **Validation Accuracy:** ~98%  
- **Stable loss & accuracy curves**  
- **Strong generalization on test images**

---

## 📊 Evaluation Metrics

- **Accuracy & Loss curves**
- **Confusion Matrix**
- **Classification Report**
  - Precision  
  - Recall  
  - F1-score  

All results validated that **MobileNetV2** is the top-performing model for deployment.

---

## 💾 Model & Class Mapping

- Best model saved as `.keras` file  
- Class names saved in `class_names.json`  
- JSON ensures correct mapping between model output (numbers) and species names

---

## 🌐 Streamlit Deployment

### Features:
- Upload any fish image (JPG/PNG)
- Image is resized + normalized
- Model predicts species
- Confidence score visualized with progress bar
- Probability JSON displayed

### Run Streamlit:
```bash
streamlit run app.py

📁 Project Structure

📦 Fish-Classification
 ┣ 📂 models/                  # trained model
 ┣ 📂 notebooks/               # training notebooks
 ┣ 📂 streamlit/               # app.py
 ┣ 📜 class_names.json
 ┣ 📜 requirements.txt
 ┣ 📜 README.md



