########################
#라이브러리 사용
import tensorflow as tf

########################
#샘플 이미지셋 불러오기

(mnist_x, mnist_y),_ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape, mnist_y.shape)

(cifar_x, cifar_y), _ = tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape, cifar_y.shape)

#########################
#이미지 출력하기
print(mnist_y[0:10])
import matplotlib.pyplot as plt
plt.imshow(mnist_x[4], cmap='gray')

print(cifar_y[0:10])
import matplotlib.pyplot as plt
plt.imshow(cifar_x[0])

##########################
# 차원 확인
import numpy as np
d1 = np.array([1,2,3,4,5,])
print(d1.shape)

d2= np.array([d1,d1,d1,d1])
print(d2.shape)

d3 = np.array([d2,d2,d2])
print(d3.shape)

d4 = np.array([d3,d3])
print(d4.shape)

########################
#(5,),(5,1),(1,5) 비교

x1 = np.array([1,2,3,4,5])
print(x1.shape)
print(mnist_y[0:5])
print(mnist_y[0:5].shape)

x2 = np.array([[1,2,3,4,5]])
print(x2.shape)

x3 = np.array([[1],[2],[3],[4],[5]])
print(x3.shape)
print(cifar_y[0:5])
print(cifar_y[0:5].shape)