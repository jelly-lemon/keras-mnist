import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# 一些参数
batch_size = 128
epochs = 1
num_classes = 10

# 获取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 归一化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 改变数据形状，格式为(n_samples, vector)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 控制台打印输出样本数量信息
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 样本标签转化为one-hot编码格式
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 创建MLP模型
model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784, )))
model.add(Dropout(rate=0.2))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()     # 在控制台输出模型参数信息
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 预测
n = 5   # 给出需要预测的图片数量，为了方便，只取前5张图片
predicted_number = model.predict(x_test[:n], n)

# 画图
plt.figure(figsize=(10, 3))
for i in range(n):
    plt.subplot(1, n, i + 1)
    t = x_test[i].reshape(28, 28)   # 向量需要reshape为矩阵
    plt.imshow(t, cmap='gray')      # 以灰度图显示
    plt.subplots_adjust(wspace=2)   # 调整子图间的间距，挨太紧了不好看
    # 第一个数字是真实标签，第二个数字是预测数值
    # 如果预测正确，绿色显示，否则红色显示
    # 预测结果是one-hot编码，需要转化为数字
    if y_test[i].argmax() == predicted_number[i].argmax():
        plt.title('%d,%d' % (y_test[i].argmax(), predicted_number[i].argmax()), color='green')
    else:
        plt.title('%d,%d' % (y_test[i].argmax(), predicted_number[i].argmax()), color='red')
    plt.xticks([])  # 取消x轴刻度
    plt.yticks([])
plt.show()
