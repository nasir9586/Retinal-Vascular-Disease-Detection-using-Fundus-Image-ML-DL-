# Diabetic Retinopathy Detection

## Introduction
Diabetic Retinopathy (DR) is a leading cause of blindness worldwide. This project aims to automate the detection of Diabetic Retinopathy using deep learning techniques. The **InceptionV3** model is utilized to classify fundus images into five categories:

- **No DR**
- **Mild NPDR** (Non-Proliferative Diabetic Retinopathy)
- **Moderate NPDR**
- **Severe NPDR**
- **PDR** (Proliferative Diabetic Retinopathy)

This deep learning-based approach enhances early detection and classification, which is crucial for timely medical intervention.

## Objective
The primary goal of this project is to develop an efficient deep learning model that can automatically classify retinal images into different DR severity levels. The model is trained using **InceptionV3** and fine-tuned to improve performance.

### Key Objectives:
- Preprocess and augment the dataset for improved model generalization.
- Implement **Transfer Learning** using InceptionV3.
- Train and evaluate the deep learning model for accurate DR detection.
- Deploy the model using **Flask** and **Gradio** for easy accessibility.

## Dataset
The dataset consists of fundus images categorized into five classes:

- **No DR**
- **Mild NPDR**
- **Moderate NPDR**
- **Severe NPDR**
- **PDR**

### Data Preprocessing:
- Images are resized to **299x299 pixels** to match **InceptionV3** input size.
- Data augmentation techniques such as **rotation, zooming, and flipping** are applied to increase model robustness.
- Labels are **one-hot encoded** for classification.

## Model Architecture
The model is built using **Transfer Learning** with **InceptionV3** as the base model.

### Layers:
- **InceptionV3 Base Model** (Pre-trained on **ImageNet**, initially frozen)
- **Global Average Pooling Layer**
- **Flatten Layer**
- **Dense Layers** (1024 → 512 → 256 neurons with **ReLU activation**)
- **Batch Normalization and Dropout (0.2)**
- **Output Layer** (**Softmax** for 5 classes)

## Training & Evaluation
- The model is trained using **Categorical Crossentropy Loss** and **Adam Optimizer**.
- Performance is evaluated using **Accuracy, Precision, Recall, and F1-score**.
- Training and validation curves are analyzed to monitor overfitting.

## Deployment
The trained model is deployed using **Flask** and **Gradio** for user-friendly accessibility.

### Deployment Steps:
1. Save the trained model in **.h5** format.
2. Create a Flask API for real-time predictions.
3. Integrate **Gradio** for an interactive web-based interface.

## Results & Insights
- Achieved **93% accuracy** on the test dataset.
- Successfully classified DR severity levels with high precision.
- Model insights help in early detection and medical intervention.

## Future Enhancements
- Implement **Ensemble Learning** for improved accuracy.
- Optimize model inference time for real-time applications.
- Extend deployment to a cloud-based API for scalability.
