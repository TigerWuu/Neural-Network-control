import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

dataset = 1000

yp = np.zeros(dataset+1)
u = np.zeros(dataset)
x_test = np.zeros((dataset-1,3))
y_test = np.zeros((dataset-1,1))
yp[0] = 0
yp[1] = 0

for k in range(1,dataset-1):
    u[k] = np.random.random_sample()*4-2
    yp[k+1] = (yp[k]*(yp[k-1]+2)*(yp[k]+2.5))/(8.5+ yp[k]**2 + yp[k-1]**2) + u[k]
    

for i in range(dataset-1):
    x_test[i]=[u[i+1] , yp[i+1] ,yp[i]] 
    y_test[i]=yp[i+2]

model = keras.models.load_model("simulation_1")
score = model.evaluate(x_test , y_test)
prediction = model.predict(x_test[:])

# plt.ion()  ## continuously show the image
# for i in range(dataset):
plt.plot(yp , "b")
plt.plot(prediction, "r")
plt.title("simulation_result")
plt.text(500 , 7, f"loss : {score[0]}" , verticalalignment="top" , horizontalalignment="center")
plt.show()
# plt.pause(0.0005)



