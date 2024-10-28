import os
import sys
import numpy as np
import tensorflow as tf

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <image_path>')
    sys.exit(1)

# Path to the saved keras model
MODEL_PATH = '../backend/models/8_dogs.keras'

# Image dimensions for the model
IMG_HEIGHT = 160
IMG_WIDTH = 160

# Path to the class names file, one class per line
CLASSNAMES_PATH = '../backend/models/8_dogs_classes.txt'

# Path to the image to predict
IMAGE_PATH = sys.argv[1]

image_path = os.path.expanduser(IMAGE_PATH)
image = tf.keras.preprocessing.image.load_img(
    image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
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

classnames = open(CLASSNAMES_PATH).read().splitlines()

model_path = os.path.expanduser(MODEL_PATH)
model = tf.keras.models.load_model(model_path)
predictions = model.predict(image_batch)

top_10_indices = np.argsort(predictions[0])[-10:][::-1]
top_10_classes = [(classnames[i], predictions[0][i]) for i in top_10_indices]

print("Top 10 predicted classes:")
for class_name, probability in top_10_classes:
    print(f'{class_name}: {probability:.4f}')
