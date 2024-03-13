# Import necessary libraries

import cv2
import numpy as np
from keras_preprocessing import image
import tensorflow as tf

# Constants for emotion labels and color mapping
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgusted',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprised'
}

COLOR_MAPPING = {
    'Angry': (0, 0, 255),        # Red
    'Disgusted': (0, 255, 0),    # Green
    'Fear': (255, 0, 0),          # Blue
    'Happy': (0, 255, 255),      # Yellow
    'Neutral': (255, 255, 0),    # Cyan
    'Sad': (255, 0, 255),        # Magenta
    'Surprised': (255, 165, 0)   # Orange
}

def initialize_emotion_model():
    
    # Initialize the emotion detection model and load pre-trained weights.

    # Returns:
    # - model (tf.keras.models.Sequential): Pre-trained emotion detection model.
    
    model = tf.keras.models.Sequential([
        # Convolutional Neural Network layers for emotion detection
         #Ideal inputshape
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
    model.load_weights('model.h5')
    return model

def preprocess_face_image(gray_img, x, y, w, h):
        
    #     Preprocess the face image for emotion prediction.

    #     Parameters:
    #    - gray_img (numpy.ndarray): Grayscale image containing the face.
    #    - x, y, w, h (int): Coordinates and dimensions of the detected face.

    #    Returns:
    #    - img_pixels (numpy.ndarray): Processed face image ready for emotion prediction.
        
    roi_gray = gray_img[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    return img_pixels

def detect_emotion(face_casc, img, gray_img, emotion_model):
    #    
    #       Detect faces and emotions in the provided image.

   	#   Parameters:
   	#   - face_casc (cv2.CascadeClassifier): OpenCV Cascade Classifier for face detection.
    #       - img (numpy.ndarray): Original image.
    #       - gray_img (numpy.ndarray): Grayscale version of the original image.
    #       - emotion_model (tf.keras.models.Sequential): Pre-trained emotion detection model.

    #       Returns:
    #       None (Displays emotion detection results in the provided image.)
    #     
    faces = face_casc.detectMultiScale(gray_img, 1.3, 5)
    for (x, y, w, h) in faces:
        
        # Extract face region and make emotion prediction
        face_image = preprocess_face_image(gray_img, x, y, w, h)

        # Make emotion prediction
        prediction = emotion_model.predict(face_image)
        max_index = int(np.argmax(prediction))
        predicted_emotion = EMOTION_LABELS[max_index]

        # Determine color based on predicted emotion
        rectangle_color = COLOR_MAPPING.get(predicted_emotion, (255, 255, 255))  # Default: White

        # Draw rectangle around the detected face with emotion-specific color
        cv2.rectangle(img, (x, y-50), (x+w, y+h+10), rectangle_color, 2)

        # Display predicted emotion on the image
        cv2.putText(img, predicted_emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def main():
    #     
    # Main function to capture video feed and perform emotion detection.

    # Usage:
    # - Run the script to capture video feed and visualize real-time emotion detection.
    # 
    # Open the camera and load Haar cascade for face detection
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    face_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize emotion detection model
    emotion_model = initialize_emotion_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert frame to grayscale and detect emotions
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect_emotion(face_casc, frame, gray_img, emotion_model)

        # Resize the frame for display and show the video feed
        resized_img = cv2.resize(frame, (1000, 700))
        cv2.imshow('Recognizer', resized_img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Run the main application
        main()
    except Exception as e:
        print(f"Error: {e}")
