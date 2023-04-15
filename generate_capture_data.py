import tensorflow as tf
import numpy as np

image_file = r"./experimental/data/dhs-logo.jpg"

image = tf.io.read_file(image_file)
image = tf.image.decode_jpeg(image)
image = tf.cast(image, tf.float32)
image = image / 255.

output_file = r"./experimental/data/captures/dhs-logo.npy"
with open(output_file, 'wb') as f:
    np.save(f, image.numpy())
