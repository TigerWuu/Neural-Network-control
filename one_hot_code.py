import numpy as np
from tensorflow import keras
import matplotlib

data = np.random.random_sample((1000, 10)) ## 1000*100 array
labels = np.random.randint(5, size=(1000, 1)) ## 0~9 1000*1 array

# 将标签转换为分类的 one-hot 编码
one_hot_labels = keras.utils.to_categorical(labels, num_classes=5)
print(one_hot_labels)