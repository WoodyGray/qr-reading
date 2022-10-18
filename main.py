import glob

import tensorflow as tf

classes = 3
colors = ['black', 'red', 'blue']

sample_size = (256, 256)
output_size = (1080, 1920)

def load_images(image, mask):
    #загрузка и преобразование в формат
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

    #разделение по каналам
    for i in range(classes):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

    #формирование бинарных многоканальных изображений
    masks = tf.stack(masks, axis=2)
    masks = tf.reshape(masks, output_size + (classes,))

    return image, masks

def augmentate_image(image, masks):
    #аугментация данных
    random_crop = tf.random.uniform((), 0.3, 1)
    #извлечение центрального фрагмента со случайным размером масштаба
    image = tf.image.central_crop(image, random_crop)
    masks = tf.image.central_crop(masks, random_crop)

    #случайное отражение изображения по горизонтали
    random_flip = tf.random.uniform((), 0, 1)
    if random_flip >= 0.5:
        image = tf.image.flip_left_right(image)
        masks = tf.image.flip_left_right(masks)

    #выходное изображение
    image = tf.image.resize(image, sample_size)
    masks = tf.image.resize(masks, sample_size)

    return image, masks

def input_layer():
    return tf.keras.layers.Input(shape=sample_size + (3,))

def downsample_block(filters, size, batch_norm=True):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Secuential()

    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernal_initializer=initializer, use_bias=False))

images = sorted(glob.glob(''))
masks = sorted(glob.glob('masks_machine\*.png'))

#формирование набора данных
images_dataset = tf.data.Dataset.from_tensor_slices(images)
masks_dataset = tf.data.Dataset.from_tensor_slices(masks)

#объединение для параллельной обработки
dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

#загрузка данных в память
dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.repeat(50) #увеличени
dataset = dataset.map(augmentate_image, num_parallel_calls=tf.data.AUTOTUNE) #аугментация

#разделение данных и их кэширование
train_dataset = dataset.take(2000).cache()
test_dataset = dataset.skip(2000).take(100).cache()

#установка размера пакета для обучения нейронки
train_dataset = train_dataset.batch(16)
test_dataset = test_dataset.batch(16)



