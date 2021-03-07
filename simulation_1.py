import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


num = 1000

yp_train = np.zeros(num+1)
u_train = np.zeros(num)
x_train = np.zeros((num-1,3))
y_train = np.zeros((num-1,1))
yp_train[0] = 0
yp_train[1] = 0


for k in range(1,num-1):
    u_train[k] = np.random.random_sample()*4-2
    yp_train[k+1] = (yp_train[k]*(yp_train[k-1]+2)*(yp_train[k]+2.5))/(8.5+ yp_train[k]**2 + yp_train[k-1]**2) + u_train[k]
    

for i in range(num-1):
    x_train[i]=[u_train[i+1] , yp_train[i+1] ,yp_train[i]] 
    y_train[i]=yp_train[i+2]

model = keras.models.Sequential([
    keras.Input(shape=3),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
mse = keras.losses.MeanSquaredError()
sgd = keras.optimizers.SGD()
model.compile(loss=mse , optimizer=sgd , metrics= ["accuracy"])

his = model.fit(x_train , y_train , epochs=40 ,batch_size = 2)
plt.plot(his.history['loss'])
plt.show()

model.save("simulation_1")





