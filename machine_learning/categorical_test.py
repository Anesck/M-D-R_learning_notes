import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn import linear_model as lin
from sklearn.preprocessing import PolynomialFeatures as Poly
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.preprocessing import OneHotEncoder

from keras import initializers, optimizers
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras import backend as K

def my_loss(y_true, y_pred):
    #loss_func = tf.keras.losses.CategoricalCrossentropy()
    #return loss_func(y_true, y_pred)
    return K.categorical_crossentropy(y_true, y_pred)


#weight_init = initializers.RandomNormal(0, 0.3)
weight_init = initializers.Constant(0)
bias_init = initializers.Constant(0.1)
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init))
#model.add(Dense(10, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init))
model.add(Dense(2, activation='softmax', kernel_initializer=weight_init, bias_initializer=bias_init))
model.compile(optimizer=optimizers.SGD(0.01), loss='categorical_crossentropy', metrics=['accuracy'])
compare_model = clone_model(model)
compare_model.compile(optimizer=optimizers.SGD(0.01), loss='categorical_crossentropy', metrics=['accuracy'])

x1 = np.linspace(-1, 1, 10).reshape(-1, 1)
x2 = np.linspace(-1, 1, 10).reshape(-1, 1)
x1, x2 = np.meshgrid(x1, x2)
x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
labels = np.sign(x[:, 1]).reshape(-1,1)

OHE = OneHotEncoder()
OHE.fit(labels)
y = OHE.transform(labels).toarray()
vt = np.arange(y.shape[0], dtype='float32').reshape(-1, 1)

y_predict = model.predict_on_batch(x)
crossentropy = -vt * y * np.log(y_predict)
print(np.mean(crossentropy, axis=0), np.sum(np.mean(crossentropy, axis=0)))

for i in range(1):
    loss = np.array(model.train_on_batch(x, y, sample_weight=vt.reshape(-1, )))
    #compare_loss = np.array(compare_model.train_on_batch(x, y, sample_weight=vt.reshape(-1, )))
    #loss = np.array(model.train_on_batch(x, y*vt))
    compare_loss = np.array(compare_model.train_on_batch(x, y*vt))
#print(loss)
print(model.get_weights())
print(compare_model.get_weights())

y_predict = model.predict_on_batch(x)
loss_func = tf.keras.losses.CategoricalCrossentropy()
print(loss_func(y, y_predict))

crossentropy = -y * np.log(y_predict)
print(np.mean(crossentropy, axis=0), np.sum(np.mean(crossentropy, axis=0)))
'''
model.fit(x, y, batch_size=10, epochs=100)
for i in range(100):
    y_predict = model.predict_on_batch(x[i].reshape(1, -1))
    print(y_predict, y_predict.shape, np.sum(y_predict, axis=1).reshape(-1, 1), np.random.choice([0, 1], p=y_predict.reshape(-1, ))==(labels[i]+1)/2)
#print(((labels.reshape(-1, 1)+1)/2 == y_predict.reshape(-1, 1)).all())
'''


#tensorboard = TensorBoard(log_dir='log')
#model.fit(x, y, batch_size=10, epochs=100, shuffle=True, callbacks=[tensorboard])
#model.fit(x_train, y_train, batch_size=10, epochs=500, shuffle=True)
#print('\n'.join(['%s:%s' % item for item in model.__dict__.items()]))
