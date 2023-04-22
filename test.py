import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from networks.select import select_G
from args import parse_args
import metasurface.solver as solver
import metasurface.conv as conv
import matplotlib.pyplot as plt
import datetime
import sys

sys.argv = ['', '--train_dir', '.',
            '--test_dir', '.',
            '--save_dir', '.',
            '--ckpt_dir', '.experimental/ckpt/',  # to use pre-given checkpoint
            '--ckpt_dir', '../gdrive/MyDrive/model_saves/princeton/', # meant for colab usage
            '--real_psf', './experimental/data/psf/psf.npy',
            '--psf_mode', 'REAL_PSF',
            '--conv_mode', 'REAL',
            '--conv', 'full_size']
args = parse_args()

# Initialize and restore deconvolution method
params = solver.initialize_params(args)
params['conv_fn'] = conv.convolution_tf(params, args)
params['deconv_fn'] = conv.deconvolution_tf(params, args)

snr = tf.Variable(args.snr_init, dtype=tf.float32)
G = select_G(params, args)
checkpoint = tf.train.Checkpoint(G=G, snr=snr)

status = checkpoint.restore(tf.train.latest_checkpoint(
    args.ckpt_dir, latest_filename=None))
status.expect_partial()

## PERFORMING DECONVOLUTION
# Check that the dimensions agree with experimental captures
assert (params['image_width'] == 720)
assert (params['psf_width'] == 360)
assert (params['network_width'] == 1080)

# Load in experimentally measured PSFs
psf = (np.load('./experimental/data/psf/psf.npy'))
psf = tf.constant(psf)
psf = tf.image.resize_with_crop_or_pad(psf, params['psf_width'], params['psf_width'])
psf = psf / tf.reduce_sum(psf, axis=(1,2), keepdims=True)


def reconstruct(img_name, psf, snr, G):
    img = np.load(img_name)
    _, G_img, _ = params['deconv_fn'](img, psf, snr, G, training=False)
    G_img_ = G_img.numpy()[0, :, :, :]

    # Vignette Correct
    vig_factor = np.load('experimental/data/vignette_factor.npy')[0, :, :, :]
    G_img_ = G_img_ * vig_factor

    # Gain
    G_img_ = G_img_ * 1.2
    G_img_[G_img_ > 1.0] = 1.0

    # Contrast Normalization
    minval = np.percentile(G_img_, 5)
    maxval = np.percentile(G_img_, 95)
    G_img_ = np.clip(G_img_, minval, maxval)
    G_img_ = (G_img_ - minval) / (maxval - minval)
    G_img_[G_img_ > 1.0] = 1.0

    plt.figure(figsize=(6, 6))
    plt.imshow(G_img_)
    today = datetime.datetime.now()
    date = today.strftime('%d%b%Y-%H_%M')
    plt.savefig('../gdrive/MyDrive/model_saves/princeton/figure/' + date + "-" +
                img_name.replace("./experimental/data/captures/", "").replace(".npy", ".png"))


# Figure 3
reconstruct('./experimental/data/captures/dhs-logo.npy', psf, snr, G)
reconstruct('./experimental/data/captures/dhs-logo-up.npy', psf, snr, G)
reconstruct('./experimental/data/captures/110802.npy', psf, snr, G)
