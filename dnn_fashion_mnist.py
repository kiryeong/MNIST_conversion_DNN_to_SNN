# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:09:56 2022

@author: kiryeong nam
"""

# 딥러닝 관련 라이브러리 불러오기
import tensorflow as tf
from tensorflow import keras # tensorflow를 쉽게 사용할 수 있도록 도와주는 라이브러리
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import backend as K

# Fashion MNIST 데이터셋 불러오기
fashion_mnist = keras.datasets.fashion_mnist

# Fashion MNIST 데이터셋 학습용(x,y), 테스트용(x,y)로 나누기
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 학습용 데이터 형태 보기
x_train.shape

# 학습용 첫 번째 데이터 보기
x_train[0]
y_train[0]

# 데이터 전처리(0 ~ 1 사이 숫자로) - 딥러닝 모델값을 입력할 때는 0~1, -1~1 까지 숫자가 들어왔을 때 모델의 성능이 좋아진다.
x_train = x_train / 255
x_test = x_test / 255

# 데이터 전처리 결과 확인
x_train[0]

N1 = 784
N2 = 100
N3 = 10

# 모델 만들기 : 입력층(784) - 은닉층(100) - 출력층(10)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28,28)), # 입력 이미지의 크기가 28X28 이므로 이를 1차원 텐서로 펼치는 것                                          
    keras.layers.Dense(N2, activation = 'relu', use_bias=False),# 784개의 값을 입력받아 100개의 값으로 인코딩, 활성함수로 relu 사용
    keras.layers.Dense(N3, activation = 'softmax', use_bias=False) # 확률을 출력해준다
])


# 모델 컴파일 : 최적화 함수, 손실 함수 설정 + 평가 지표 설정 + 가중치 초기화
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()

# 모델 학습 : 전체 데이터는 10번 반복
model.fit(x_train, y_train, epochs = 10, batch_size = 100)

#모델 평가
results = model.evaluate(x_test, y_test)
print('test loss, test acc:', results)

