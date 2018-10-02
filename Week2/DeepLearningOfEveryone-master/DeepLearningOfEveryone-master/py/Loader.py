# !/usr/bin/python
# -*- coding: UTF-8 -*-

# 引入模块
from keras.datasets import mnist  # 引入MNIST数据集
from keras.utils import to_categorical
# keras.utils.to_categorical函数是把类别标签转换为onehot编码（一种方便计算机处理的二元编码）
from keras import backend as K  # 引入Keras的后端，在这里是Tensorflow，之后叫做K

img_rows, img_cols = 28, 28  # 图像高和宽为28*28像素

def loading():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # 输入数据并预处理

    '''在如何表示一组彩色图片的问题上，Theano和TensorFlow发生分歧
       因此，keras输入数据有两种格式，一种是通道数放在前面，一种是通道数放在后面
       此处有个坑：卷积核与所使用的后端不匹配，不会报错，因为它们的shape是完全一致的'''
    if K.image_data_format() == 'channels_first':
        '''channels_first（通道维靠前）：
           Theano模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式（100,3,16,32），Caffe也是如此；
           第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数,后面两个是高和宽了'''
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        '''channels_last（通道维靠后）：本训练用的是这个！
           TensorFlow表达形式是（100,16,32,3）'''
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # 转换输入数据类型
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # 对数据进行归一化到0-1,图像数据最大是255
    X_train /= 255
    X_test /= 255

    # 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中
    y_test, y_train = to_categorical(y_test, 10), to_categorical(y_train, 10)

    return input_shape, (X_train, y_train), (X_test, y_test)  # 返回数据

