
#  Handwritten Digits Recognition 

##  Project Overview

This project aims to build a **deep learning model** that can accurately recognize handwritten digits (0–9) from image data.
It uses the **MNIST dataset**, a widely used benchmark in computer vision, which contains **70,000 grayscale images** of handwritten digits.

The primary objective is to explore **image preprocessing, neural networks, and evaluation metrics** while developing a system that achieves high accuracy in digit classification.

---

##  Objectives

* Understand and preprocess handwritten digit image data
* Implement machine learning and deep learning models
* Train and evaluate models on benchmark data
* Visualize predictions to interpret model behavior
* Provide a baseline system that could be extended for real-world digit recognition (e.g., bank cheques, postal codes, forms)

---

##  Dataset

* **Dataset Name:** MNIST Handwritten Digits
* **Source:** Keras / Yann LeCun’s dataset repository
* **Data Format:** 28×28 grayscale images (flattened to 784 features for dense models, or 28×28×1 for CNNs)
* **Size:**

  * **Training Set:** 60,000 images
  * **Test Set:** 10,000 images
* **Labels:** Digits (0–9), total **10 classes**

---

## 🛠 Technologies & Tools

* **Programming Language:** Python 
* **Libraries & Frameworks:**

  * `NumPy`, `Pandas` → Data manipulation
  * `Matplotlib`, `Seaborn` → Visualization
  * `TensorFlow / Keras` → Deep Learning models
  * `Scikit-learn` → Model evaluation, metrics, train-test split

---

##  Workflow

### 1. Data Loading & Inspection

* Import MNIST dataset from Keras
* Check dataset shape and sample images

### 2. Data Preprocessing

* **Normalization:** Scale pixel values (0–255) → (0–1)
* **Reshaping:** (28×28) → (28×28×1) for CNN input
* **One-Hot Encoding:** Convert labels (e.g., “5”) into categorical vectors

### 3. Model Development

* **Baseline Model:** Simple Dense Neural Network (MLP)
* **Advanced Model:** Convolutional Neural Network (CNN)

  * Conv2D layers → Extract spatial features
  * MaxPooling2D → Downsample image
  * Flatten → Convert 2D features into vector
  * Dense layers → Learn digit patterns
  * Output layer (Softmax) → Predict probability for each digit

### 4. Model Training

* Optimizer: **Adam**
* Loss Function: **Categorical Crossentropy**
* Metrics: **Accuracy**
* Epochs: 10–20 (depending on performance)
* Validation split used to prevent overfitting

### 5. Evaluation & Results

* Test set (10,000 images) used for evaluation
* Metrics: Accuracy, Confusion Matrix
* Plotted **accuracy vs. epochs** and **loss vs. epochs**

### 6. Visualization

* Display random images with predicted vs. actual labels
* Show **confusion matrix heatmap** to identify common misclassifications

---

##  Results

* The **Convolutional Neural Network (CNN)** achieved:

  * **Test Accuracy:** \~98%
  * **Low Loss** on validation and test sets
* Visualizations confirmed strong model performance, with only occasional errors (commonly between similar digits like “4” & “9”).

---

##  Conclusion

This project demonstrates how **deep learning (CNNs)** can effectively solve image recognition problems.
The model generalizes well to unseen handwritten digits, achieving near-human accuracy.

 Key Takeaways:

* Normalization and preprocessing are critical for image models
* CNNs outperform traditional ML models on image data
* The project provides a solid baseline for real-world handwriting recognition applications

---

##  Future Improvements

* Experiment with deeper CNN architectures (ResNet, EfficientNet)
* Apply **data augmentation** (rotation, shifting, zoom) for robustness
* Use **transfer learning** with pre-trained models
* Deploy the model as a **REST API** for real-time digit recognition
* Build a **web interface** for user-uploaded digit recognition

---

##  Acknowledgments

This project was developed as part of the **Capstone Project (PRCP-1002)** for Data Science learning.
Special thanks to the **MNIST dataset creators** and open-source libraries used in this work.


