import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


# define some constants
num_classes = 3
batch_size = 40
epochs = 81

# loading/setting up data
x_train = np.load('x_train_01_try1.npy')[:-10, :, :, :]
y_train = np.load('y_train_01_try1.npy')[:-10]

print(len(x_train))
print(len(y_train))

x_train = np.concatenate((x_train,np.load('x_train_01_try2.npy')[:-10, :, :, :]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_01_try2.npy')[:-10]), axis = 0)
print(len(x_train))
print(len(y_train))


x_train = np.concatenate((x_train,np.load('x_train_01_try3.npy')[:-10, :, :, :]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_01_try3.npy')[:-10]), axis = 0)
print(len(x_train))
print(len(y_train))


x_train = np.concatenate((x_train,np.load('x_train_02_try1.npy')[:-10, :, :, :]), axis = 0)
x_train = np.concatenate((x_train,np.load('x_train_02_try3.npy')[:-10, :, :, :]), axis = 0)

x_train = np.concatenate((x_train,np.load('x_train_03_try1.npy')[:-10, :, :, :]), axis = 0)
x_train = np.concatenate((x_train,np.load('x_train_03_try2.npy')[:-10, :, :, :]), axis = 0)
x_train = np.concatenate((x_train,np.load('x_train_03_try3.npy')[:-10, :, :, :]), axis = 0)

x_train = np.concatenate((x_train,np.load('x_train_04_try1.npy')[:-10, :, :, :]), axis = 0)
x_train = np.concatenate((x_train,np.load('x_train_04_try2.npy')[:-10, :, :, :]), axis = 0)
x_train = np.concatenate((x_train,np.load('x_train_04_try3.npy')[:-10, :, :, :]), axis = 0)

x_train = np.concatenate((x_train,np.load('x_train_05_try1.npy')[:-10, :, :, :]), axis = 0)
x_train = np.concatenate((x_train,np.load('x_train_05_try2.npy')[:-10, :, :, :]), axis = 0)
x_train = np.concatenate((x_train,np.load('x_train_05_try3.npy')[:-10, :, :, :]), axis = 0)

x_train = np.concatenate((x_train,np.load('x_train_06_try1.npy')[:-10, :, :, :]), axis = 0)
x_train = np.concatenate((x_train,np.load('x_train_06_try2.npy')[:-10, :, :, :]), axis = 0)
x_train = np.concatenate((x_train,np.load('x_train_06_try3.npy')[:-10, :, :, :]), axis = 0)


y_train = np.concatenate((y_train,np.load('y_train_02_try1.npy')[:-10]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_02_try3.npy')[:-10]), axis = 0)

y_train = np.concatenate((y_train,np.load('y_train_03_try1.npy')[:-10]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_03_try2.npy')[:-10]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_03_try3.npy')[:-10]), axis = 0)

y_train = np.concatenate((y_train,np.load('y_train_04_try1.npy')[:-10]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_04_try2.npy')[:-10]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_04_try3.npy')[:-10]), axis = 0)

y_train = np.concatenate((y_train,np.load('y_train_05_try1.npy')[:-10]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_05_try2.npy')[:-10]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_05_try3.npy')[:-10]), axis = 0)

y_train = np.concatenate((y_train,np.load('y_train_06_try1.npy')[:-10]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_06_try2.npy')[:-10]), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_06_try3.npy')[:-10]), axis = 0)

print("Length of x_train")
print(len(x_train))
print("Length of y_train")
print(len(y_train))


x_train = x_train.reshape(x_train.shape[0], 300*600*3)
x_train = x_train.astype('float32')
x_train /= 255

print(x_train.shape[0], 'train samples' + '\n')

y_train = keras.utils.to_categorical(y_train, num_classes)

# build the model
model = Sequential()
model.add(Dense(32, activation='sigmoid', input_shape=(300*600*3,)))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr = 0.005),
              metrics=['accuracy'])

# train the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=None)

# save the model
model.save("self_driving_car_model.h5")
print("Model saved to disk.")
