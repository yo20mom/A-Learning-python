###########################
# 이미지 읽어서 데이터 준비하기
import glob
import tensorflow as tf
from tensorflow.keras import utils, layers, models

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

paths = glob.glob('C:\\Users\\SKPLANET\\Downloads\\machine_image\\notMNIST_small\\*\\*.png')
paths = np.random.permutation(paths)
독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])
종속 = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])
print(독립.shape, 종속.shape)

독립 = 독립.reshape(18724, 28, 28, 1)
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)

################################
# 모델을 완성합니다.
X = layers.Input(shape=[28, 28, 1])

H = layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
H = layers.MaxPool2D()(H)

H = layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = layers.MaxPool2D()(H)

H = layers.Flatten()(H)
H = layers.Dense(120, activation='swish')(H)
H = layers.Dense(84, activation='swish')(H)
Y = layers.Dense(10, activation='softmax')(H)

model = models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

#########################
# 모델을 학습
# Ctrl + shift + j
history = model.fit(독립, 종속, epochs=10)

############################
# 모델을 이용합니다.
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)
pd.DataFrame(종속[0:5]).round(2)

model.summary()
