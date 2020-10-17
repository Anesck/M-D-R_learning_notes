import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

from keras.utils.np_utils import to_categorical

input_layer = Input(shape=(2, ))
hidden_layer = Dense(10, activation='relu')(input_layer)
output_mu = Dense(1, activation='tanh')(hidden_layer)
output_sigma = Dense(1, activation='softplus')(hidden_layer)
output_layer = Concatenate()([output_mu, output_sigma])
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=optimizers.Adam(), loss='mse')
model.summary()
exit()

x1 = np.linspace(-1, 1, 10).reshape(-1, 1)
x2 = np.linspace(-1, 1, 10).reshape(-1, 1)
x1, x2 = np.meshgrid(x1, x2)
x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))

labels = np.sign(x[:, 1]).reshape(-1,1)
y = to_categorical((labels+1)/2, 2)

model.fit(x, y, batch_size=32, epochs=500)
#model.fit(x, [y[:, 0], y[:, 1]], batch_size=32, epochs=500)
output = model.predict_on_batch(x)
#print(output[:50, ], output[50:, ])
np.set_printoptions(precision=5, suppress=True)
print("{}".format(output))
