import tensorflow as tf

classes = 3
colors = ['black', 'red', 'blue']

sample_size = (256, 256)
output_size = (1080, 1920)

def load_images(image, mask):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, output_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0

    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, output_size)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    masks = []

    for i in range(classes):
        masks.append(tf.where(tf.equal))

