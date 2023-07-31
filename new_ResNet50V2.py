import tensorflow as tf
from stack2 import stack2


def new_ResNet50V2(include_top=True,
                   preact=True,
                   use_bias=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax'
                   ):
    img_input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if not preact:
        x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack2(x, 64, 3, name='conv2')
    x = stack2(x, 128, 4, name='conv3')
    x = stack2(x, 256, 6, name='conv4')
    x = stack2(x, 512, 3, stride1=1, name='conv5')

    if preact:
        x = tf.keras.layers.BatchNormalization(name='post_bn')(x)
        x = tf.keras.layers.Activation('relu', name='post_relu')(x)
    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = tf.keras.layers.Dense(classes, activation=classifier_activation, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)

    model = tf.keras.Model(img_input, x, name='new_resnet50v2')

    # 自動下載 ImageNet 的預訓練權重
    if weights == 'imagenet':
        if tf.io.gfile.exists('resnet50v2_weights_tf_dim_ordering_tf_kernels.h5'):
            model.load_weights('resnet50v2_weights_tf_dim_ordering_tf_kernels.h5')
        else:
            raise ValueError(
                "Cannot find the local weight file 'resnet50v2_weights_tf_dim_ordering_tf_kernels.h5'. Please specify "
                "the correct file path.")

        if tf.io.gfile.exists('resnet50_imagenet_1000.h5'):
            model.load_weights('resnet50_imagenet_1000.h5')
        else:
            raise ValueError(
                "Cannot find the local weight file 'resnet50_imagenet_1000.h5'. Please specify the correct file path.")

    return model
