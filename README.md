# Skin Disease Classification using Deep Learning 🩺

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-96.7%25-brightgreen?style=for-the-badge)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-blueviolet?style=for-the-badge)


A powerful deep learning model for classifying 24 different types of skin diseases from images. This project leverages transfer learning with the **Xception** architecture, fine-tuned to achieve high accuracy in dermatological diagnosis.



---

## 📋 Table of Contents
* [About The Project](#about-the-project)
* [Model Architecture](#model-architecture)
* [Dataset](#dataset)
* [Installation](#installation-️)
* [Usage](#usage-)
* [Results](#results-)
* [Contributing](#contributing-)
* [License](#license-)
* [Acknowledgments](#acknowledgments-)

---

## 📝 About The Project

The goal of this project is to build and train a robust image classification model capable of distinguishing between various skin conditions. Early and accurate diagnosis of skin diseases can be critical, and this project explores the application of state-of-the-art computer vision techniques to aid in this process.

This implementation uses:
* **TensorFlow & Keras** for building and training the deep learning model.
* **Transfer Learning** with the pre-trained **Xception model** on ImageNet to leverage its powerful feature extraction capabilities.
* **Data Augmentation** to create a more robust model that generalizes well to new, unseen images.
* **Callbacks** like `EarlyStopping` and `ModelCheckpoint` for efficient training and to save the best-performing model.

---

## 🧠 Model Architecture

The model is built upon the **Xception** convolutional neural network. The core methodology is **transfer learning**:

1.  **Base Model**: The pre-trained Xception model is used as a fixed feature extractor. Its convolutional base, trained on the massive ImageNet dataset, is highly effective at identifying low-level and mid-level features like edges, textures, and patterns in images. The top classification layer of Xception is removed.
2.  **Freezing Layers**: The weights of the base model's layers are frozen to prevent them from being updated during the initial training phase, thus preserving the learned ImageNet features.
3.  **Custom Classifier Head**: A custom classification head is added on top of the base model. This head consists of:
    * A `GlobalAveragePooling2D` layer to reduce the spatial dimensions.
    * A `Dropout` layer (with a rate of 0.5) to prevent overfitting.
    * A `Dense` layer with 1024 neurons and a `ReLU` activation function.
    * A final `Dense` output layer with 24 neurons (one for each skin disease class) and a `softmax` activation function for multi-class classification.



---

## 📂 Dataset

The model was trained on the **Skin Disease Dataset**, which contains images across 24 distinct classes of skin conditions. As trained data is private collection of clinical images for skin diseases, which cannot be made public due to patient privacy and confidentiality concerns.

### Using Your Own Dataset
While the original dataset is private, the code is structured to work with any image dataset that follows the same directory format. To use your own data, please structure it as follows:
```text
/path/to/your/dataset/
├── class_1/
│   ├── image_001.jpg
│   ├── image_002.png
│   └── ...
├── class_2/
│   ├── image_003.jpg
│   ├── image_004.jpg
│   └── ...
└── ...
    └── class_24/
        ├── image_n.jpg
        └── ...
```
Each subdirectory should be named after a specific skin condition (e.g., `Acne`, `Eczema`), and it should contain all the corresponding images for that class.

* **Data Split**: The dataset is split into three sets:
    * **Training Set**: 80%
    * **Validation Set**: 10%
    * **Testing Set**: 10%
* **Augmentation**: To prevent overfitting and improve generalization, the training data is augmented in real-time with the following transformations:
    * Random rotations
    * Width and height shifts
    * Shear and zoom transformations
    * Horizontal flips

---
## 🚀 Getting Started

This project is contained entirely within a single Jupyter Notebook. You can get started in a couple of ways.

### Prerequisites
You'll need to have the following Python libraries installed. You can install them all with a single command.
* `tensorflow`
* `tensorflow_hub`
* `pandas`
* `scikit-learn`
* `matplotlib` (for visualizing results)

### Option 1: Running on Your Local Machine

1.  **Download the Notebook**
    Click the `Download raw file` button at the top of the file viewer on GitHub to save the `.ipynb` file to your computer.

2.  **Install Dependencies**
    Open your terminal or command prompt and run the following command to install all necessary libraries:
    ```sh
    pip install tensorflow pandas scikit-learn matplotlib
    ```

3.  **Launch Jupyter Notebook**
    Navigate to the directory where you saved the file and start Jupyter:
    ```sh
    jupyter notebook
    ```
    Then, open the downloaded `.ipynb` file from the Jupyter interface in your browser.

### Option 2: Running on Google Colab (Recommended)

This is the easiest way to get started, as it requires no local setup and provides free access to a GPU.

1.  **Open in Colab**
    Click the "Open in Colab" badge at the top of this README.

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/YOUR_NOTEBOOK_NAME.ipynb)

2.  **Run the Cells**
    Once the notebook is open in Colab, you can run each code cell sequentially by pressing `Shift + Enter`. Most dependencies are pre-installed in Colab.

## ▶️ Usage

The notebook is organized into sequential cells. To train the model and see the results, simply run all the cells from top to bottom.

1.  **Set Your Data Path**: In the designated code cell, update the `data_dir` variable to point to the location of your dataset.
    ```python
    # Update this line with the path to your dataset
    data_dir = '/path/to/your/dataset'
    ```
2.  **Execute the Notebook**: Run each cell in order. The notebook will handle data loading, model building, training, and evaluation.
3.  **View Results**: The final cells will output the model's test accuracy and save the trained model files (`best_model.h5`, etc.) to your environment.
