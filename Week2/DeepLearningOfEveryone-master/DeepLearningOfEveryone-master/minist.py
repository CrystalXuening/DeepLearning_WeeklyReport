# !/usr/bin/python
# -*- coding: UTF-8 -*-

# 引入模块
import os  # os模块包含普遍的操作系统功能
import numpy as np  # 是一个科学计算的库，提供了矩阵运算的功能
from libDL import Loader, Saver  # 加载数据和保存的模块
from keras.models import Sequential  # 序贯模型，一个按顺序建立的模型，先加一个层，再加一个层这种
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
# Conv2D是二维卷积层；Dense代表是全连接层；MaxPooling2D层为空域信号施加最大值池化；
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不影响结果，似乎是优化CPU的匹配

# 加载训练数据，分为训练和测试用的
input_shape, (X_train, y_train), (X_test, y_test) = Loader.loading()

# 构建神经网络
model = Sequential()  # 创建一个序贯模型，或者说是开始创建一个神经网络
# 加上二维卷积层为输入层，32个卷积通道（输出），卷积核窗口选用3*3像素窗口，选用relu激活函数
# 第一次需要给出input_shape，后面默认输入为前一层输出
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(Flatten())  # 加上一个Flatten层，把多维的输入一维化
model.add(Dense(10, activation='softmax'))  # 加一个全连接层,10个输出维度，选用softmax激活函数
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
# 编译创建好的模型，对网络的学习过程进行配置
# 用loss来计算误差，优化器选用RMSprop（该优化器通常是面对递归神经网络时的一个良好选择）
# Metrices是性能评估，只是作为评价网络表现的一种“指标”，为了直观地了解算法的效果，充当view的作用，并不参与到优化过程
model.summary()  # 打印出模型概况

# training
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 保存预测数据
session_data = np.load('./predict/session-10-27.npy')
session_predict = model.predict(session_data)
Saver.save(session_predict)



