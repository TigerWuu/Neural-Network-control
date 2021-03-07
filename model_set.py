from tensorflow import keras

## method 1
model1 = keras.models.Sequential([
  keras.Input(shape=(784,)),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

## method 2
model2 = keras.models.Sequential()
model2.add(keras.Input(shape=(784,)))
model2.add(keras.layers.Dense(64, activation='relu'))
model2.add(keras.layers.Dense(64, activation='relu'))
model2.add(keras.layers.Dense(10, activation='softmax'))

## method 3
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(64, activation="relu")(inputs)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10,activation="softmax")(x)
model3 = keras.Model(inputs=inputs, outputs=outputs, name="test_model")

model1.summary()
model2.summary()
model3.summary()

