# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:35:53 2020

@author: user
"""


# 딥러닝 관련 라이브러리 불러오기
import tensorflow as tf
from tensorflow import keras # tensorflow를 쉽게 사용할 수 있도록 도와주는 라이브러리
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터셋 불러오기
mnist = keras.datasets.mnist

# MNIST 데이터셋 학습용(x,y), 테스트용(x,y)로 나누기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train_one_hot = tf.one_hot(y_train,depth = 10)

train1_idx = (np.argmax(y_train_one_hot, 1) == 1) 
train2_idx = (np.argmax(y_train_one_hot, 1) == 2) 
train3_idx = (np.argmax(y_train_one_hot, 1) == 3) 
train4_idx = (np.argmax(y_train_one_hot, 1) == 4) 
train5_idx = (np.argmax(y_train_one_hot, 1) == 5) 
train6_idx = (np.argmax(y_train_one_hot, 1) == 6) 
train7_idx = (np.argmax(y_train_one_hot, 1) == 7) 
train8_idx = (np.argmax(y_train_one_hot, 1) == 8) 
train9_idx = (np.argmax(y_train_one_hot, 1) == 9)
train0_idx = (np.argmax(y_train_one_hot, 1) == 0)

train1_imgs = x_train[train1_idx] # 1 train 이미지 추출
train1_labels = y_train_one_hot[train1_idx] # 1 train 라벨 추출
train2_imgs = x_train[train2_idx]
train2_labels = y_train_one_hot[train2_idx]
train3_imgs = x_train[train3_idx]
train3_labels = y_train_one_hot[train3_idx]
train4_imgs = x_train[train4_idx]
train4_labels = y_train_one_hot[train4_idx]
train5_imgs = x_train[train5_idx]
train5_labels = y_train_one_hot[train5_idx]
train6_imgs = x_train[train6_idx]
train6_labels = y_train_one_hot[train6_idx]
train7_imgs = x_train[train7_idx]
train7_labels = y_train_one_hot[train7_idx]
train8_imgs = x_train[train8_idx]
train8_labels = y_train_one_hot[train8_idx]
train9_imgs = x_train[train9_idx]
train9_labels = y_train_one_hot[train9_idx]
train0_imgs = x_train[train0_idx]
train0_labels = y_train_one_hot[train0_idx]

n_train1 = train1_imgs.shape[0]
print("The number of training(1) images : {}, shape : {}".format(n_train1, train1_imgs.shape))
n_train2 = train2_imgs.shape[0]
print("The number of training(2) images : {}, shape : {}".format(n_train2, train2_imgs.shape))
n_train3 = train3_imgs.shape[0]
print("The number of training(3) images : {}, shape : {}".format(n_train3, train3_imgs.shape))
n_train4 = train4_imgs.shape[0]
print("The number of training(4) images : {}, shape : {}".format(n_train4, train4_imgs.shape))
n_train5 = train5_imgs.shape[0]
print("The number of training(5) images : {}, shape : {}".format(n_train5, train5_imgs.shape))
n_train6 = train6_imgs.shape[0]
print("The number of training(6) images : {}, shape : {}".format(n_train6, train6_imgs.shape))
n_train7 = train7_imgs.shape[0]
print("The number of training(7) images : {}, shape : {}".format(n_train7, train7_imgs.shape))
n_train8 = train8_imgs.shape[0]
print("The number of training(8) images : {}, shape : {}".format(n_train8, train8_imgs.shape))
n_train9 = train9_imgs.shape[0]
print("The number of training(9) images : {}, shape : {}".format(n_train9, train9_imgs.shape))
n_train0 = train0_imgs.shape[0]
print("The number of training(0) images : {}, shape : {}".format(n_train0, train0_imgs.shape))

image1_res03=train1_imgs.sum(axis=2)
image1 = image1_res03.sum(axis=1)
plt.hist(image1,bins = 100)

image2_res03=train2_imgs.sum(axis=2)
image2 = image2_res03.sum(axis=1)
plt.hist(image2,bins = 100)

image3_res03=train3_imgs.sum(axis=2)
image3 = image3_res03.sum(axis=1)
plt.hist(image3,bins = 100)

image4_res03=train4_imgs.sum(axis=2)
image4 = image4_res03.sum(axis=1)
plt.hist(image4,bins = 100)

image5_res03=train5_imgs.sum(axis=2)
image5 = image5_res03.sum(axis=1)
plt.hist(image5,bins = 100)

image6_res03=train6_imgs.sum(axis=2)
image6 = image6_res03.sum(axis=1)
plt.hist(image6,bins = 100)

image7_res03=train7_imgs.sum(axis=2)
image7 = image7_res03.sum(axis=1)
plt.hist(image7,bins = 100)

image8_res03=train8_imgs.sum(axis=2)
image8 = image8_res03.sum(axis=1)
plt.hist(image8,bins = 100)

image9_res03=train9_imgs.sum(axis=2)
image9 = image9_res03.sum(axis=1)
plt.hist(image9,bins = 100)

image0_res03=train0_imgs.sum(axis=2)
image0 = image0_res03.sum(axis=1)
plt.hist(image0,bins = 100)



