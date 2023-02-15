# testcode
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# 作弊代码 tensorflow2 转 1
# hello = tf.constant('hello,tensorflow')
# sess = tf.compat.v1.Session()
# print(sess.run(hello))
# import os
# import tensorflow as tf
# from tensorflow import keras
#
#
# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#
# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28, 28)))
# model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
#
# # 训练
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_images, train_labels, epochs=50)
#
# # 测试
# model.evaluate(test_images, test_labels)
#
# # 判断单张图片的类型
# # 0 T-shirt/Top; 1 Trouser/Pants; 2 Pullover shirt; 3 Dress; 4 Coat; 5 Sandal; 6 Shirt; 7 Sneaker; 8 Bag; 9 Ankle boot
# model.predict((test_images[0]).reshape(1, 28, 28))
# # 选最大值即可
# import numpy as np
# print(np.argmax(model.predict((test_images[0]).reshape(1, 28, 28))))
#
# # 检验
# print(test_labels[0])
# GCN 的代码

