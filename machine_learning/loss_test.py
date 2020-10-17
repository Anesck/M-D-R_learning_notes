import gym
import numpy as np
import matplotlib.pyplot as plt
import keras

from sklearn import linear_model as lin
from sklearn.preprocessing import PolynomialFeatures as Poly
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.preprocessing import OneHotEncoder

from keras import initializers, optimizers
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda
from keras import backend as K

def myloss(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    a = y_true[:, 0]
    b = y_true[:, 1]
    c = y_pred[:, 0]
    d = y_pred[:, 1]
    #y_true = y_true - y_pred + y_pred
    return K.mean(a + b + c + d)

weight_init = initializers.RandomNormal(0, 0.1)
bias_init = initializers.Constant(0)
input_layer = Input(shape=(2, ))
hidden_layer = Dense(10, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init)(input_layer)
hidden_layer_mu = Dense(1, activation='tanh', kernel_initializer=weight_init, bias_initializer=bias_init)(hidden_layer)
hidden_layer_sigma = Dense(1, activation='tanh', kernel_initializer=weight_init, bias_initializer=bias_init)(hidden_layer)
output_mu = Lambda(lambda x: x*2)(hidden_layer_mu)
output_sigma = Lambda(lambda x: x+1)(hidden_layer_sigma)
output_layer = Concatenate()([output_mu, output_sigma])
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=optimizers.Adam(), loss='mse')

x1 = np.linspace(-1, 1, 10).reshape(-1, 1)
x2 = np.linspace(-1, 1, 10).reshape(-1, 1)
x1, x2 = np.meshgrid(x1, x2)
x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
labels = np.sign(x[:, 1]).reshape(-1,1)

OHE = OneHotEncoder()
OHE.fit(labels)
y = OHE.transform(labels).toarray()
y = np.ones_like(y)
vt = np.arange(y.shape[0], dtype='float32').reshape(-1, )*1000

model.fit(x, y, batch_size=32, epochs=100, sample_weight=vt)
#model.train_on_batch(x[0].reshape(-1, 2), y[0].reshape(-1, 2), sample_weight=np.array([2]))
print(model.predict_on_batch(x))
