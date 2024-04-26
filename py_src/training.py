import tensorflow as tf
from net import get_model, IMAGE_SIZE, BATCH_SIZE
import os

train_dir = "../dataset/cats_and_dogs_256x256/train/"
eval_dir = "../dataset/cats_and_dogs_256x256/val/"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # rescale the image to [0.0, 1.0] for better training
eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # binary classification
)

eval_generator = eval_datagen.flow_from_directory(
    eval_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
)

model = get_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

train_steps_per_epoch = train_generator.samples // BATCH_SIZE
val_steps_per_epoch = eval_generator.samples // BATCH_SIZE
model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=100, # arbitrary, the training will stop when the validation accuracy is not improving
    validation_data=eval_generator,
    validation_steps=val_steps_per_epoch,
    # use early stopping to stopping to get the weights when the validation accuracy is the best
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy", mode="max")]
)

os.makedirs("../models", exist_ok=True)
model.save("../model/cats_dogs_net.h5")