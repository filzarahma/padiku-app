# Rice Leaf Disease Detection with ResNet-50

This project demonstrates the application of deep learning to classify rice leaf diseases using the ResNet-50 architecture. It employs transfer learning and data augmentation techniques to achieve high accuracy in detecting diseases across four classes.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)
7. [Use Case](#use-case)
8. [How to Use](#how-to-use)
9. [Dependencies](#dependencies)
10. [Acknowledgments](#acknowledgments)

---

## Project Overview

Rice is a staple crop, and its diseases significantly impact food security worldwide. This project addresses the challenge of rice leaf disease classification using a Convolutional Neural Network (CNN). By leveraging the ResNet-50 model, this project classifies images into one of four disease categories:

1. **Bacterial Blight**
2. **Blast**
3. **Brown Spot**
4. **Tungro**

---

## Dataset Description

The dataset consists of images of rice leaves categorized into four classes of diseases. The dataset is split as follows:

-   **Training Set**: 3,082 images
-   **Validation Set**: 2,052 images (split from the training set)
-   **Testing Set**: 800 images

Image dimensions are resized to **224x224** pixels to fit the ResNet-50 input requirements.

---

## Model Architecture

This project uses a pre-trained **ResNet-50** model as the feature extractor, followed by a custom classification head. The detailed architecture includes:

1. **Input Layer**: `(224, 224, 3)`
2. **ResNet-50 Feature Extractor**: Pre-trained on ImageNet.
3. **Global Average Pooling**: Reduces dimensionality.
4. **Dropout Layer**: Prevents overfitting.
5. **Dense Layers**:
    - Fully connected layer with 128 neurons (ReLU activation).
    - Output layer with 4 neurons (linear activation for multi-class classification).

Optimizer: `Stochastic Gradient Descent (SGD)`  
Loss Function: `Squared Hinge Loss`

---

## Training Process

Key aspects of the training process include:

1. **Image Augmentation**:
    - Rotation up to 90 degrees.
    - Horizontal and vertical flipping.
    - Nearest neighbor fill for augmented pixels.
2. **Validation Split**: 40% of the training set is used for validation.
3. **Callback**: Training stops early if the model achieves >90% accuracy and validation accuracy.

---

## Evaluation Metrics

The model is evaluated using the following metrics:

-   **Accuracy**: Overall prediction correctness.
-   **Precision**: Correctly identified positives.
-   **Recall**: Coverage of actual positives.
-   **F1-Score**: Harmonic mean of precision and recall.
-   **Confusion Matrix**: Visualizes classification results.

---

## Results

### Training and Validation Loss

![Training and Validation Loss](/docs/screenshots/TrainingAndValidationLoss.png)

### Training and Validation Accuracy

![Training and Validation Accuracy](/docs/screenshots/TrainingAndValidationAccuracy.png)

### Confusion Matrix

![Confusion Matrix](/docs/screenshots/ConfusionMatrix.png)

### Final Metrics

-   **Training Accuracy**: 100%
-   **Validation Accuracy**: 95.86%
-   **Test Accuracy**: 98.37%

### Classification Report

| Class            | Precision  | Recall     | F1-Score   | Support |
| ---------------- | ---------- | ---------- | ---------- | ------- |
| Bacterial Blight | 1.0000     | 0.9450     | 0.9717     | 200     |
| Blast            | 0.9851     | 0.9950     | 0.9900     | 200     |
| Brown Spot       | 0.9950     | 0.9950     | 0.9950     | 200     |
| Tungro           | 0.9569     | 1.0000     | 0.9780     | 200     |
| **Overall**      | **0.9838** | **0.9838** | **0.9837** | **800** |

---

## Use Case

This repository is designed for researchers, agronomists, and developers looking for a robust solution to detect and classify rice leaf diseases, contributing to better crop management and food security.

---

## How to Use

1. Clone this repository:
    ```
    git clone https://github.com/filzarahma/padiku-app.git
    ```
2. Install the required dependencies:
    ```
    pip install tensorflow==2.8.0 keras==2.8.0 scikit-learn matplotlib seaborn
    ```
3. Upload the dataset into the appropriate directories:
    - `train/`
    - `test/`
4. Run the Jupyter Notebook:
    ```
    jupyter notebook rice_leaf_disease_detection.ipynb
    ```
5. Train the model or use the pre-trained weights `model_CS.h5`.

---

## Dependencies

The project was built and tested using the following dependencies:

-   TensorFlow 2.8.0
-   Keras 2.8.0
-   scikit-learn
-   matplotlib
-   seaborn

Ensure you have Python 3.7 or higher installed.

---

## Acknowledgments

This project is made possible with:

-   The **Rice Leaf Disease Images Dataset**.
-   The ResNet-50 architecture for its robust feature extraction capabilities.

If you use this project, please cite or reference appropriately.
