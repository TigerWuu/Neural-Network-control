import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


x = np.zeros(101)
u = np.zeros(100)
x[0] = 0.1

x_train = np.zeros((100,2))
y_train = np.zeros((100,1))


for k in range(100):
    u[k] = np.sin(2*np.pi*k/25) + np.sin(2*np.pi*k/10)
    x[k+1] = x[k]/((x[k])**2) + (u[k])**3

plt.plot(u)
plt.plot(x)
plt.show()

for i in range(100):
    x_train[i] = [x[i] , u[i]]
    y_train[i] = x[i+1]

model = keras.models.Sequential([
    keras.Input(shape=2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    #keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

mse = keras.losses.MeanSquaredError()
sgd = keras.optimizers.SGD()
model.compile(loss=mse , optimizer=sgd , metrics= ["accuracy"])

his = model.fit(x_train , y_train , epochs=600 ,batch_size = 16)
plt.plot(his.history['loss'])
plt.show()

prediction = model.predict(x_train)
plt.plot(x[1:])
plt.plot(prediction)
plt.show()

model.save("simulation_2")

