import tensorflow as tf

BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)

def get_model():
    return tf.keras.models.Sequential(
    [
        tf.keras.layers.ZeroPadding2D(padding=(0, 1), input_shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='valid'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.ZeroPadding2D(padding=(0, 1)),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='valid'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)