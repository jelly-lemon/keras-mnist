import keras
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

# 一些参数
batch_size = 128
epochs = 1
num_classes = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)   # 输入数据形状

# 获取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 归一化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 改变数据形状，格式为(n_samples, rows, cols, channels)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# 控制台打印输出样本数量信息
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 样本标签转化为one-hot编码格式
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 创建CNN模型
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()     # 在控制台输出模型参数信息
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
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
