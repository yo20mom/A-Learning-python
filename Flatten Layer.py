##Flatten layer를 활용한 이미지 학습
import tensorflow as tf
import pandas as pd
#데이터를 준비하고
##with reshape
(독립, 종속),_ = tf.keras.datasets.mnist.load_data()
#print(독립.shape, 종속.shape)
독립 = 독립.reshape(60000, 784)
종속 = pd.get_dummies(종속)
#print(독립.shape, 종속.shape)

#모델을 만들고
X = tf.keras.layers.Input(shape=[784])
H = tf.keras.layers.Dense(84, activation='swish')(X)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

#모델을 학습하고
model.fit(독립, 종속, epochs=10)

#모델을 이용합니다.
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)

#데이터를 준비하고
(독립, 종속),_ = tf.keras.datasets.mnist.load_data()
print(독립.shape, 종속.shape)
#독립 = 독립.reshape(60000, 784)
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)

#모델을 만들고
X = tf.keras.layers.Input(shape=[28, 28])
H = tf.keras.layers.Flatten()(X)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

#모델을 학습하고
model.fit(독립, 종속, epochs=5)

#모델을 이용합니다.
pred = model.predict(독립[0:5])
pd.DataFrame(pred).round(2)




