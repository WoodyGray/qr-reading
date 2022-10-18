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
    #задает входной слой нейронной сети и устанавливает размер входных данных
    return tf.keras.layers.Input(shape=sample_size + (3,))

#описание блоков которые формирует анкодер
def downsample_block(filters, size, batch_norm=True):
    #метод инициализации весовых коф-ов
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Secuential()
    #включение сверточного слоя
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernal_initializer=initializer, use_bias=False))
    #добавление слоя пакетной нормализации
    if batch_norm:
        resul.add(tf.keras.layers.LeakyReLU())
    #установка активационной функции
    result.add(tf.keras.layers.LeakyReLU())
    return result
#формировка декодера нейронки
def upsample_block(filters, size, dropout=False):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()

    #анти свертка
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, pading='same',
                                        kernal_initialiser=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    #возможность добавления dropout слоев
    if dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.ReLU())
    return result

#задает выходной слой
def output_layer(size):
    initialiser = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv2DTranspose(CLASSES, size, strides=2, padding='same',
                                           kernel_initializer=initialiser, activation='sigmoid')

#оценка точности работы нейронки
def dice_mc_metric(a, b):
    #распаковка многоканальной маски
    a = tf.unstack(a, axis=3)
    b = tf.unstack(b, axis=3)

    dice_summ = 0

    #нахождения средне значение коф-та dice для всех объектов
    for i, (aa, bb) in enumerate(zip(a, b)):
        numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
        denomerator = tf.math.reduce_sum(aa + bb) + 1
        dice_sum += numenator / denomerator

    avg_dice = dice_summ / classes

    return avg_dice

#функия потерь
def dice_mc_loss(a, b):
    return 1 - dice_mc_metric(a, b)

#уменьшение коф-та dice
def dice_bce_mc_loss(a, b):
    return 0.3 * dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)

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

inp_layer = input_layer()

#энкодер
downsample_stack = [
    downsample_block(64, 4, batch_norm=False),
    downsample_block(128, 4),
    downsample_block(256, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
]

#декодер
upsample_stack = [
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4)
]

out_layer = output_layer(4)

x = inp_layer

downsample_skips = []

#соединение блоков енкодера и декодера
for block in downsample_stack:
    x = block(x)
    downsample_skips.append(x)

downsample_skips = reversed(downsample_skips[:-1])

for up_block, down_block in zip(upsample_stack, downsample_skips):
    x = up_block(x)
    #реализация межслоевых соединений(конкотинация)
    x = tf.keras.layers.Concatenate()([x, down_block])

out_layer = out_layer(x)

#модель в которой указываются входные и выходные слои
unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)

#компиляция модели с функциями потерь и метрикой(алгоритм адам)
unet_like.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_loss])

history_dice = unet_like.fit(train_dataset, validation_data=test_dataset, epochs=25, initial_epoch=0)

unet_like.save_weights('путь сохранения')
