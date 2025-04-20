Apologies for any confusion earlier. I understand now that you're seeking a complete `README.md` file structured exactly like your provided example, with all sections and code blocks included. Here's the comprehensive `README.md` content tailored for the [eyop/FER](https://github.com/eyop/FER) project:

---

```markdown
# Real-time Facial Expression Recognition User Manual

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [Overview](#11-overview)
   - 1.2 [Key Features](#12-key-features)
2. [Getting Started](#2-getting-started)
   - 2.1 [Installation](#21-installation)
   - 2.2 [Configuration](#22-configuration)
3. [Recognition](#3-recognition)
   - 3.1 [Running Recognition](#31-running-recognition)
4. [CNN Model Training](#4-cnn-model-training)
   - 4.1 [Training Your Model](#41-training-your-model)
5. [Usage Tips](#5-usage-tips)
6. [Dependencies](#6-dependencies)

---

## 1. Introduction

### 1.1 Overview

Real-time Facial Expression Recognition is an emotion recognition project that combines computer vision and deep learning to identify emotions from facial expressions. This user manual provides a comprehensive guide on how to install, configure, and utilize the system for real-time emotion recognition and CNN model training.

This project consists of two main components: Emotion Recognition and CNN Model Training. The Emotion Recognition component utilizes a trained Convolutional Neural Network (CNN) to detect faces in real-time and predict the associated emotion. The CNN Model Training component involves training a CNN model on a dataset of facial expressions to recognize various emotions.

### 1.2 Key Features

- Real-time emotion prediction using a pre-trained Convolutional Neural Network (CNN).
- Face detection with bounding boxes for enhanced visualization.
- Color-coded emotions for easy identification.
- Option to train your own CNN model for emotion recognition.

---

## 2. Getting Started

### 2.1 Installation

To install Real-time Facial Expression Recognition, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/eyop/FER.git
   ```

2. Or download the .zip file and extract the content:

   ```bash
   gh repo clone eyop/FER
   ```

### 2.2 Configuration

1. Navigate to the project directory:

   ```bash
   cd FER
   ```

2. Configure the dataset directories for model training in `Train.py`:

   ```python
   train_dir = 'data/train'
   val_dir = 'data/test'
   ```

---

## 3. Recognition

### 3.1 Running Recognition

1. Run the Emotion Recognition component:

   ```bash
   python3 Gui.py
   ```

   This will launch the application and start real-time facial expression recognition using your webcam.

---

## 4. CNN Model Training

### 4.1 Training Your Model

1. Run the CNN Model Training component:

   ```bash
   python3 Train.py
   ```

   To train the model, you'll need data. For this project, the FER-2013 dataset is used. However, you're free to use any dataset suitable for the task. Ensure your data is properly formatted and check for any potential biases. You can download the FER-2013 dataset [here](https://www.kaggle.com/msambare/fer2013). Import the directories with data into your project folder and reference them within Python.

   We'll have two directories: one for training data and another for validation. The validation directory will be the 'test' folder. If the names don't suit you, feel free to change the directory names to something else. Just ensure they are recognizable for others.

---

## 5. Usage Tips

1. Run emotion recognition:

   ```bash
   python3 Gui.py
   ```

2. Explore the CNN Model Training component:

   ```bash
   python3 Train.py
   ```

3. Press `Q` to quit the application and close the window.

---

## 6. Dependencies

To install these modules, check their documentation:

- [Python 3.8](https://www.python.org/downloads/)
- [OpenCV](https://docs.opencv.org/master/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [Keras](https://keras.io/)
- [Pillow](https://pypi.org/project/Pillow/)
- [FER-2013](https://www.kaggle.com/msambare/fer2013)

Use the following command to install these modules:

```bash
pip install opencv-python numpy tensorflow keras matplotlib pillow
```

---

## 📩 Contact & Contributions

Contributions, feedback, and improvements are welcome! Feel free to fork the project and submit a pull request.

For queries, suggestions, or collaboration:

- **GitHub**: [@eyop](https://github.com/eyop)

---

> 🎉 Thank you for checking out this project! If you found it helpful, don't forget to ⭐ star the repo and share it with others.
```

---

Feel free to copy and paste this content directly into your `README.md` file. If you need further customization or additional sections, don't hesitate to ask! 
