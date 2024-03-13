# Real-time Facial Expression Recognition User Manual

## Table of Contents

1. [Introduction](#introduction)
   - 1.1 [Overview](#overview)
   - 1.2 [Key Features](#key-features)
2. [Getting Started](#getting-started)
   - 2.1 [Installation](#installation)
   - 2.2 [Configuration](#configuration)
3. [Recognition](#Recognition)
   - 3.1 [Running_Recognition](#3-recognition)
4. [CNN Model Training](#4-cnn-model-training)
   - 4.1 [Training Your Model](#training-your-model)

5. [Usage Tips](#5-usage-tips)

## Dependencies
> To install these modules, check their documentation.
- [Python 3.8](https://www.python.org/downloads/)
- [OpenCV](https://docs.opencv.org/master/)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](https://numpy.org/)
- [Keras](https://keras.io/)
- [Pillow](https://pypi.org/project/Pillow/)
- [FER-2013](https://www.kaggle.com/msambare/fer2013)

# use this To install these modules
  ```bash
     pip install opencv-python numpy tensorflow keras matplotlib
  ```
## 1. Introduction

### 1.1 Overview

Real-time Facial Expression Recognition is an emotion recognition project that combines computer vision and deep learning to identify emotions from facial expressions. This user manual provides a comprehensive guide on how to install, configure, and utilize EmoDetect for real-time emotion recognition and CNN model training.

This project consists of two main components: Emotion Recognition and CNN Model Training. The Emotion Recognition component utilizes a trained Convolutional Neural Network (CNN) to detect faces in real-time and predict the associated emotion. The CNN Model Training component involves training a CNN model on a dataset of facial expressions to recognize various emotions.

### 1.2 Key Features

- Real-time emotion prediction using a pre-trained Convolutional Neural Network (CNN).
- Face detection with bounding boxes for enhanced visualization.
- Color-coded emotions for easy identification.
- Option to train your own CNN model for emotion recognition.

## 2. Getting Started

### 2.1 Installation

To install Real-time Facial Expression Recognition, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/eyop/FER.git
   
   ```
2. Download the .zip file and extract the content:
   ```bash
     $ gh repo clone eyop/FER
   ```
### 2.2 Configuration
1. Navigate to the project directory

   ```bash
   cd FER
   ```4
2. configure the dataset directories for model training in the Train.py
   ```python
     train_dir = 'data/train'
     val_dir = 'data/test'
   ```
## 3. Recognition
### 1. Running Recognition
1. Run the Emotion Recognition component:
   - To use the application:
   ```bash
   python3 Gui.py
   ```
## 4. CNN Model Training
1. Run the CNN Model Training component:
   - To train the model:
   ```bash
   python3 Train.py
   ```

To train our model, we'll need data. For our project we used FER-2013 as our dataset. But you are free to use any data suitable for the task. Just make sure you fit the coding by your type of images and check you data for any potential biases. You can download the FER-2013 dataset [here](https://www.kaggle.com/msambare/fer2013). Import the directories with data into your project folder and call them within Python.

We'll have two directories: one for test data and another for validation. The validation directory will be the 'test' folder. Don't get confused by the names. If the name doesn't suit you, feel free to change the directory names to something else. Just make sure it will be recognizable for someone else.

## 5. Usage Tips

1. Run emotion recognition:
   ```bash
   python3 Gui.py
   ```
2. Explore the CNN Model Training component:
   ```bash
   python Train.py
   ```

3. Press `Q` to quit the application and close the window.




