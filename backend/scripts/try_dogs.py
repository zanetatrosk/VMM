import os
import numpy as np
import tensorflow as tf
from PIL import Image

batch_size = 32
IMG_HEIGHT = 160
IMG_WIDTH = 160



CLASSNAMES_PATH_16_DOGS = './models/16_dogs_v1_classes.txt'
CLASSNAMES_PATH_8_DOGS = './models/8_dogs_v2_classes.txt'
MODEL_PATH_16_DOGS = './models/16_dogs_v1.keras'
MODEL_PATH_8_DOGS = './models/8_dogs_v2.keras'
IMAGE_PATH = 'tmp.jpg'

def getPictureRecognition(imgInput: Image, pickedModel: str):
    if pickedModel == '16_dogs':
        model_path = MODEL_PATH_16_DOGS
        class_name_path = CLASSNAMES_PATH_16_DOGS
    elif pickedModel == '8_dogs':
        model_path = MODEL_PATH_8_DOGS
        class_name_path = CLASSNAMES_PATH_8_DOGS
    else:
        return 'Invalid model'
        
    # Define the path to the image and the model
    results = []

    # load image into file
    imgInput.save(IMAGE_PATH)
    
    classnames = open(os.path.expanduser(class_name_path)).read().splitlines()
    print(classnames)
    
    image = tf.keras.preprocessing.image.load_img(
        IMAGE_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    image_batch = tf.expand_dims(
        tf.keras.preprocessing.image.img_to_array(image), 0
    )

    image_preprocess = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    image_batch = image_preprocess.flow(
        tf.expand_dims(tf.keras.preprocessing.image.img_to_array(image), 0)
    )

    model = tf.keras.models.load_model(os.path.expanduser(model_path))
    predictions = model.predict(image_batch)

    top_10_indices = np.argsort(predictions[0])[-10:][::-1]
    top_10_classes = [(classnames[i], predictions[0][i]) for i in top_10_indices]

    print("Top 10 predicted classes:")
    for class_name, probability in top_10_classes:
        results.append((class_name, probability))
        print(f'{class_name}: {probability:.4f}')
    return results