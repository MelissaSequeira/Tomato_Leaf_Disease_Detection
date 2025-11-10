# 🍅 Tomato Leaf Disease Prediction using CNN

A deep learning project that classifies tomato leaf diseases using Convolutional Neural Networks (CNN).  
The model is trained on the PlantVillage Tomato dataset and predicts 10 different tomato diseases.

---

## 📌 Overview

This project focuses on identifying tomato leaf diseases using computer vision.  
A CNN model was trained using augmented image data, with preprocessing such as rescaling, normalization, and label encoding.

Key features:
- End-to-end dataset preparation  
- CNN model building, training, and evaluation  
- Metrics visualization (accuracy, loss, confusion matrix)  
- Prediction examples  

---

## 📂 Dataset

**Source:** Kaggle (PlantVillage Dataset)

**Number of Classes: 10**
- Tomato Bacterial Spot  
- Tomato Early Blight  
- Tomato Late Blight  
- Tomato Leaf Mold  
- Tomato Septoria Leaf Spot  
- Tomato Spider Mites  
- Tomato Target Spot  
- Tomato Yellow Leaf Curl Virus  
- Tomato Mosaic Virus  
- Tomato Healthy  

**Dataset Split:**
- **70% — Training**
- **20% — Validation**
- **10% — Testing**

**Image Format:** RGB images (various sizes, resized to 128×128)

---

## 🔧 Preprocessing Techniques

### ✅ **Rescaling (1./255)**
- Converts pixel range from **0–255 → 0–1**
- Helps faster convergence
- Prevents exploding gradients  

---

### ✅ **Data Augmentation**
Used to increase dataset diversity:
- Rotation
- Zoom
- Horizontal Flip  

This helps prevent overfitting and improves generalization.

---

### ✅ **Label Encoding**
- Converts disease class names → integer values  
- Efficient for softmax output layer  
- Does **not** create multiple columns (unlike One-Hot Encoding)

---

### ✅ **Normalization**
- Ensures consistent pixel value distribution  
- Stabilizes training  
- Reduces noise variation  

---

## 🧠 Model Architecture

### **Baseline CNN Model**

