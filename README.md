## Tomato Leaf Disease Prediction using CNN

This project uses Convolutional Neural Networks (CNN) to classify tomato leaf diseases from the PlantVillage dataset. The model is trained on 10 different tomato leaf disease categories and includes image preprocessing, data augmentation, model development, training, evaluation, and result visualization.

---

## Project Overview
The objective of this project is to automatically detect tomato leaf diseases using computer vision. A CNN is trained on thousands of labeled tomato leaf images to classify them into one of ten disease categories. This approach supports early disease detection, helping farmers improve crop health and reduce losses.

---

## Dataset Description

**Dataset Source:** Kaggle PlantVillage Dataset  
**Dataset Used:** Only tomato-related classes were extracted.  

**Total Classes (10):**
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
- Training: 70%
- Validation: 20%
- Testing: 10%

All images were resized to **128 × 128 × 3 (RGB)** for CNN input consistency.

---

## Preprocessing Methods

### Rescaling
Pixel values from the images are rescaled from the range 0–255 to 0–1 using:
rescale = 1./255

yaml
Copy code
This improves model convergence and stabilizes training.

### Data Augmentation
To avoid overfitting:
- Random rotation
- Zoom transformation
- Horizontal flipping  
These transformations increase dataset diversity artificially.

### Label Encoding
Class names are converted to integer labels (0–9).  
This avoids creating multiple columns like one-hot encoding.

### Normalization
Ensures consistent pixel distribution across images, improving training stability.

---

## Model Architecture

### Baseline CNN
Conv2D(32, 3x3, ReLU)
MaxPooling2D(2x2)
Conv2D(64, 3x3, ReLU)
MaxPooling2D(2x2)
Flatten
Dense(128, ReLU)
Dense(10, Softmax)

yaml
Copy code

---

## Optimized CNN Model
Conv2D(64, 3x3, ReLU, input_shape=(128,128,3))
MaxPooling2D(2x2)
Conv2D(128, 3x3, ReLU)
MaxPooling2D(2x2)
Flatten
Dense(256, ReLU)
Dropout(0.5)
Dense(10, Softmax)

yaml
Copy code

**Optimizer:** Adam  
**Learning Rate:** 0.0005  
**Loss Function:** categorical_crossentropy  
**Metric:** accuracy  

---

## Training Procedure

- Epochs: 20  
- Batch size: 32  
- Early stopping/monitoring validation loss  
- Data augmentation applied only to training data  
- Validation data used to track overfitting  

Steps:
1. Load and preprocess dataset  
2. Create train/val/test splits  
3. Build CNN  
4. Train model  
5. Evaluate  
6. Save model  

---

## Evaluation Metrics

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Confusion Matrix
- Precision / Recall / F1-score (per class & weighted)
- Overall Test Accuracy
- Prediction visualization

---

## Results Summary

- Optimized model achieves higher accuracy than baseline  
- Augmentation reduces overfitting  
- Model performs well in distinguishing disease classes  
- Confusion matrix highlights misclassified classes  
- Prediction examples show correct identification of leaf diseases  

---

## Installation

Install dependencies:
pip install -r requirements.txt

shell
Copy code

### `requirements.txt`
tensorflow
keras
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib

yaml
Copy code

---

## How to Run

### Train the Model
python train.py

shell
Copy code

### Evaluate Model
python evaluate.py

shell
Copy code

### Run the Streamlit App
streamlit run app.py

yaml
Copy code

---

## Project Structure

Tomato-Leaf-Disease-Prediction/
│

├── data/

│ ├── train/

│ ├── val/

│ └── test/

│

├── models/

│ └── tomato_disease_model.h5

│

├── figures/

│ ├── accuracy_curve.png

│ ├── loss_curve.png

│ ├── confusion_matrix.png

│ └── predictions.png

│

├── train.py

├── evaluate.py

├── app.py

└── README.md


---

## Future Enhancements

- Integrate transfer learning (EfficientNet, MobileNet, ResNet)
- Implement Grad-CAM heatmap for explainability
- Deploy model using TensorFlow Lite for mobile applications
- Add real-time webcam-based disease detection
- Improve dataset variety with real farm images
- Add segmentation models for region-specific disease detection

---

## Author

**Melissa Sequeira**  
AI/ML Enthusiast | Computer Vision Project | 2025

---
