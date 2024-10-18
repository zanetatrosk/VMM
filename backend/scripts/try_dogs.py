import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import pathlib

batch_size = 32
img_height = 180
img_width = 180

def getPictureRecognition(imgInput: Image):
    print("TensorFlow version:", tf.__version__)
    # Define the path to the image and the model
    model_path = '/home/zaneta/VMM/project/saved_model/dog_model.keras'
    dataset_dir = os.path.expanduser('~/.keras/datasets/dogs')
    results = []

    # load image into file
    imgInput.save('tmp.jpg')
    img = keras.preprocessing.image.load_img(
        os.path.expanduser('tmp.jpg'), target_size=(img_height, img_width)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Load your saved model
    model = tf.keras.models.load_model(model_path)

    # Predict the content of the image
    predictions = model.predict(img_array)

    classnames = os.listdir(dataset_dir)
    # Get the top 10 predicted classes
    top_10_indices = np.argsort(predictions[0])[-10:][::-1]
    top_10_classes = [(classnames[i], predictions[0][i]) for i in top_10_indices]

    # Display the top 10 predicted classes
    print("Top 10 predicted classes:")
    for class_name, probability in top_10_classes:
        results.append(f'{class_name}: {probability:.4f}')
        print(f'{class_name}: {probability:.4f}')
    return results