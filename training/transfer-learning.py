import os
import pathlib
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# Input and output paths
DATASET_PATH = './dog_breed_dataset'
MODEL_SAVE_PATH = '../backend/models/'
MODEL_SAVE_NAME = 'transfer_learning_dog_breeds'

# Dataset selection and processing parameters
SEED = 12356
IMG_HEIGHT = 160
IMG_WIDTH = 160
VALIDATION_RATIO = 0.2

# Model hyperparameters
BATCH_SIZE = 32
DROPOUT_RATE = 0.2

# Training parameters
LEARNING_RATE = 0.001
MAX_EPOCHS = 10000
PATIENCE_EPOCHS_STOP = 8
PATIENCE_EPOCHS_REDUCE_LR = 3

### Dataset loading and preprocessing ###
dataset_dir = pathlib.Path(os.path.expanduser(DATASET_PATH))
class_names = sorted(os.listdir(dataset_dir))
n_classes = len(class_names)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_RATIO,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.1
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_RATIO
)

train_set = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    seed=SEED
)

valid_set = valid_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    seed=SEED
)

print('Number of classes:', n_classes)
print('Number of training samples:', train_set.samples)
print('Number of validation samples:', valid_set.samples)

### Saving the class names to a file ###
with open(MODEL_SAVE_PATH + MODEL_SAVE_NAME + '_classes.txt', 'w') as f:
    for class_name in class_names:
        f.write(class_name + '\n')

### Model definition and training using transfer learning ###
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH + MODEL_SAVE_NAME + '.keras',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=PATIENCE_EPOCHS_STOP,
    verbose=1)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=PATIENCE_EPOCHS_REDUCE_LR,
    min_lr=LEARNING_RATE / 10,
    verbose=1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

model.summary()

history = model.fit(
    train_set,
    validation_data=valid_set,
    callbacks=[save_callback, early_stopping_callback, reduce_lr_callback],
    epochs=MAX_EPOCHS)

### Plotting the training history ###
plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and validation accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and validation loss')

plt.show()