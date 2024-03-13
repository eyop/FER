# Import necessary libraries

import numpy as np  # Importing numpy library for numerical operations (currently not used)
import cv2  # Importing OpenCV library for computer vision tasks (currently not used)
from tensorflow.compat.v1 import ragged  # Import the compat.v1 module for RaggedTensorValue (unused duplicate import)
from keras_preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator for data augmentation
import matplotlib.pyplot as plt  # Importing matplotlib for plotting graphs and images (used for plotting training history)

# Dataset directories
train_dir = 'data/train'  # Directory path for training data
val_dir = 'data/test'  # Directory path for validation data

# Image Data generators
# Create data generators for training and validation datasets
train_datagen = ImageDataGenerator(rescale=1./255)  # Data generator for training data with pixel normalization
val_datagen = ImageDataGenerator(rescale=1./255)  # Data generator for validation data with pixel normalization

train_generator = train_datagen.flow_from_directory(
  train_dir,
  target_size=(48, 48),
  batch_size=64,
  color_mode='grayscale',
  class_mode='categorical'
)# Generating batches of augmented training data

validation_generator = val_datagen.flow_from_directory(
  val_dir,
  target_size=(48, 48),
  batch_size=64,
  color_mode='grayscale',
  class_mode='categorical'
)

# CNN Model
# Build a convolutional neural network model
model=tf.keras.models.Sequential([
    # Ideal input shape
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #Prevent overfitting
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    #Layers with 1024 neurons
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    #Dense with posible classes
    tf.keras.layers.Dense(7, activation='softmax')
])

# Function to display training history
# Plot training and validation accuracy and loss over epochs
def plot_training_history(history):
        """
    Plot training and validation accuracy and loss over epochs.

    Parameters:
    - history (tf.keras.callbacks.History): Training history.
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
# Compile, fit and save model
# Compile the model with categorical crossentropy loss, Adam optimizer, and accuracy metric

model.compile(
    loss = 'categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Fit the model to the training data and validate on the validation data

history = model.fit(
    train_generator,
    epochs=50,
    validation_data = validation_generator,
    verbose = 1
)

# Save the trained model weights to a file
model.save_weights("model.h5")