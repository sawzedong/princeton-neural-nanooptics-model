import tensorflow as tf
import numpy as np
from args import parse_args
import metasurface.solver as solver

args = parse_args()
params = solver.initialize_params(args)

input_file = r"./experimental/data/dhs-logo.jpg"
output_file = r"./experimental/data/captures/dhs-logo1.npy"

def load(image_width, image_width_padded, augment):
    # image_width = Width for image content
    # image_width_padded = Width including padding to accomodate PSF
    def load_fn(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image, tf.float32)
        image = image / 255.

        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
        image = tf.image.resize_with_crop_or_pad(
            image, image_width_padded, image_width_padded)
        
        with open(output_file, 'wb') as f:
            np.save(f, np.array([image]))

        return (image, image)  # Input and GT
    return load_fn

## replace 1080 with params['load_width']
load_fn = load(params['out_width'], 1080, augment=False)
image = load_fn(input_file)



