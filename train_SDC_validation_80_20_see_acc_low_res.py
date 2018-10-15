import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


seed = 7
# define some constants
num_classes = 3
batch_size = 40
epochs = 50

# loading/setting up data
x_train = np.load('x_train_low_res_1.npy')
y_train = np.load('y_train_low_res_1.npy')

print(len(x_train))
print(len(y_train))

x_train = np.concatenate((x_train,np.load('x_train_low_res_2.npy')), axis = 0)
y_train = np.concatenate((y_train,np.load('y_train_low_res_2.npy')), axis = 0)
# print(len(x_train))
# print(len(y_train))
#
#
# x_train = np.concatenate((x_train,np.load('x_train_low_res_3.npy')), axis = 0)
# y_train = np.concatenate((y_train,np.load('y_train_low_res_3.npy')), axis = 0)
# print(len(x_train))
# print(len(y_train))
#
# x_train = np.concatenate((x_train,np.load('x_train_low_res_4.npy')), axis = 0)
# x_train = np.concatenate((x_train,np.load('x_train_low_res_5.npy')), axis = 0)
# x_train = np.concatenate((x_train,np.load('x_train_low_res_6.npy')), axis = 0)
# x_train = np.concatenate((x_train,np.load('x_train_low_res_7.npy')), axis = 0)
# x_train = np.concatenate((x_train,np.load('x_train_low_res_8.npy')), axis = 0)
# x_train = np.concatenate((x_train,np.load('x_train_low_res_9.npy')), axis = 0)
# x_train = np.concatenate((x_train,np.load('x_train_low_res_10.npy')), axis = 0)
# x_train = np.concatenate((x_train,np.load('x_train_low_res_11.npy')), axis = 0)
#
# y_train = np.concatenate((y_train,np.load('y_train_low_res_4.npy')), axis = 0)
# y_train = np.concatenate((y_train,np.load('y_train_low_res_5.npy')), axis = 0)
# y_train = np.concatenate((y_train,np.load('y_train_low_res_6.npy')), axis = 0)
# y_train = np.concatenate((y_train,np.load('y_train_low_res_7.npy')), axis = 0)
# y_train = np.concatenate((y_train,np.load('y_train_low_res_8.npy')), axis = 0)
# y_train = np.concatenate((y_train,np.load('y_train_low_res_9.npy')), axis = 0)
# y_train = np.concatenate((y_train,np.load('y_train_low_res_10.npy')), axis = 0)
# y_train = np.concatenate((y_train,np.load('y_train_low_res_11.npy')), axis = 0)

print("Length of x_train")
print(len(x_train))
print("Length of y_train")
print(len(y_train))


x_train = x_train.reshape(x_train.shape[0], 240*320*3)
x_train = x_train.astype('float32')
x_train /= 255

print(x_train.shape[0], 'train samples' + '\n')

y_train = keras.utils.to_categorical(y_train, num_classes)



# split into 80% for train and 20% for test
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=seed)

del x_train
del y_train

# build the model
model = Sequential()
model.add(Dense(32, activation='sigmoid', input_shape=(240*320*3,)))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.0025),
              metrics=['accuracy'])

# train the model
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))

#model.fit(X_train, y_train,
#          validation_data=(X_test,y_test), epochs=150, batch_size=10)


# save the model
model.save("self_driving_car_model.h5")
print("Model saved to disk.")
