import time

import numpy as np
import pandas as pd
from tensorflow.keras import layers, models

print('Begin...', time.strftime('%D %H:%M:%S', time.localtime(time.time()))) # 开始时间

# 读取数据
img_train = pd.read_csv('train2.csv')
img_test = pd.read_csv('test.csv')

# 将原始数据转换为图片格式，利用CNN，图片格式（m，n_H，n_W，n_C）
img_train = np.array(img_train)
img_test = np.array(img_test)

train_images = np.zeros((img_train.shape[0], 28, 28, 1))
train_labels = np.zeros(img_train.shape[0])
test_images = np.zeros((img_test.shape[0], 28, 28, 1))
test_lables = np.zeros(img_test.shape[0])

for i in range(img_train.shape[0]):
    train_images[i] = img_train[i][1:].reshape(28, 28, 1)
    train_labels[i] = img_train[i][0].astype(int)
for i in range(img_test.shape[0]):
    test_images[i] = img_test[i].reshape(28, 28, 1)

# Normalization
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 将整数序列编码为二进制矩阵
def victorize_sequences(sequences, dimension=10):
    results = np.zeros((len(sequences), dimension), dtype='float32')
    for i, sequence in enumerate(sequences):
        results[i, int(sequence)] = 1.
    return results

train_labels = victorize_sequences(train_labels)# (42000, 0) --> (42000, 10)
test_lables = victorize_sequences(test_lables) # (28000, 0) --> (28000, 10)

print(train_images.shape) # (42000, 28, 28 ,1)
print(train_labels.shape) # (42000, 10)
print(test_images.shape) # (28000, 28, 28 ,1)
print(test_lables.shape) # (28000, 10) 目前全0

# 模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 3D 转成 1D
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, batch_size=64)
pred = model.predict(test_images, batch_size=256)  # 预测结果
print(pred.shape)
pred_X = np.argmax(pred, axis=1)

# 输出结果
result = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': pred_X})
result.to_csv("cNN_result5.csv", index=False)
 
print('End...', time.strftime('%D %H:%M:%S', time.localtime(time.time())))  # 结束时间
