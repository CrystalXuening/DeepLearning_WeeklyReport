# !/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np  # 引入numpy模块
from PIL import Image  # 引入pillow模块，基本上算是Python的图像处理标准库

image_sequence = ['a', 'b', 'c', 'd', 'e']  # 图片名称的顺序，下面的代码就按照这个顺序依次打开图像
value_sequence = []  # 用来存放下面的代码产出的数据

for code in image_sequence:
    im = Image.open('./%s.jpg' % code)  # 打开一个jpg图像文件
    im.thumbnail((28, 28))  # 缩放到28*28像素的尺寸
    value = np.array(im).reshape(28, 28, 1)  # 不改变它的数据，变成一个新的形状，1这个参数是自动计算元素个数吗？？？
    value = value / 255  # 归一化，因为图像数据最大是255
    value_sequence.append(value)  # 把数据加到value_sequence上

value_array = np.array(value_sequence)  # 存储单一数据类型的多维数组，不过我没明白这一步的意义是？
np.save('./session-10-27', value_array)  # 把数据存到session-10-27中


