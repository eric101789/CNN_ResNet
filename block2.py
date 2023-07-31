import tensorflow as tf


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    preact = tf.keras.layers.BatchNormalization(name=name + '_preact_bn')(x)
    preact = tf.keras.layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = tf.keras.layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = tf.keras.layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name+'_1_conv')(preact)
    x = tf.keras.layers.BatchNormalization(name=name+'_1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name+'_1_relu')(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+'_2_pad')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name=name+'_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'_2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name+'_2_relu')(x)

    x = tf.keras.layers.Conv2D(4*filters, 1, name=name+'_3_conv')(x)
    x = tf.keras.layers.Add(name=name+'out')([shortcut, x])
    return x
