import gym
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model as lin
from sklearn.preprocessing import PolynomialFeatures as Poly
from sklearn.preprocessing import MinMaxScaler as MMS

from keras import initializers, optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

weight_init = initializers.RandomNormal(0, 0.3)
bias_init = initializers.Constant(0.1)
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init))
model.add(Dense(10, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init))
model.add(Dense(1, kernel_initializer=weight_init, bias_initializer=bias_init))
model.compile(optimizer=optimizers.RMSprop(0.01), loss='mse')

x = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
y = np.sin(x)

mms = MMS()
x_train = mms.fit_transform(x)
y_train = mms.fit_transform(y)

#tensorboard = TensorBoard(log_dir='log')
#model.fit(x, y, batch_size=10, epochs=100, shuffle=True, callbacks=[tensorboard])
#model.fit(x_train, y_train, batch_size=10, epochs=500, shuffle=True)
#print('\n'.join(['%s:%s' % item for item in model.__dict__.items()]))
#model.fit(x[500:], y[500:], batch_size=10, epochs=100, shuffle=True)
#model.train_on_batch(x[500:], y[500:])
#model.train_on_batch(x[:500], y[:500])
for step in range(100):
    indexes = np.random.randint(1000, size=32)
    loss = model.train_on_batch(x[indexes], y[indexes])
    #loss = model.train_on_batch(x, y)
    print("loss: {} after {} step".format(loss, step+1))

    if step % 5 == 0:
        y_predict = model.predict_on_batch(x[indexes])
        
        plt.plot(x[indexes], y[indexes], '.')
        plt.plot(x[indexes], y_predict, '^')
        plt.show()

y_predict = model.predict(x)
#y_predict = mms.inverse_transform(model.predict(x_train))

X = Poly(degree=10).fit_transform(x)
Y = lin.Ridge().fit(X,y)


plt.plot(x, y, '.')
plt.plot(x, y_predict, '^')
#plt.plot(x, Y.predict(X), '+')
plt.show()

model.summary()
print(model.get_weights())
