~~~python
# Define the model
model = Sequential()

model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))# 32
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3))#  30
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(10, 1, 1, activation='relu')) #30
model.add(MaxPooling2D(pool_size=(2, 2))) #15
model.add(Convolution2D(96, 3, 3, border_mode='same')) #15
model.add(Activation('relu')) 
model.add(Convolution2D(96, 3, 3))  # 13
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(10, 1, 1, activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) #6
model.add(Convolution2D(192, 3, 3, border_mode='same')) #6
model.add(Activation('relu'))
model.add(Convolution2D(192, 3, 3)) # 4
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #2
model.add(Convolution2D(10, 2))
model.add(Flatten())
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
~~~

#### Model 1 

Dense layer was removed, Altered Dropouts, Inserted 1x1 for kernel scale up 

---



#### Model 2

~~~python
# Define the model
model = Sequential()

model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))# 32
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(48, 3, 3))#  30
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Convolution2D(10, 1, 1, activation='relu')) #30
model.add(MaxPooling2D(pool_size=(2, 2))) #15
model.add(Convolution2D(96, 3, 3, border_mode='same')) #15
model.add(Activation('relu')) 
model.add(BatchNormalization())
model.add(Convolution2D(96, 3, 3))  # 13
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Convolution2D(10, 1, 1, activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) #6
model.add(Convolution2D(192, 3, 3, border_mode='same')) #6
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(192, 3, 3)) # 4
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))# Added DropOut here
model.add(MaxPooling2D(pool_size=(2, 2))) #2
model.add(Convolution2D(10, 2))
model.add(Flatten())
model.add(Activation('softmax'))

#Added Learning Rate
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.003 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])
~~~

Still Overfitting too much and Loss is too much and Val acc caps at 75 % will add more kernels next models and add layers

---

#### Model 3



~~~Python
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(32, 32, 3)))# 32
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Convolution2D(64, 3, 3))#  30
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3,, border_mode='same'))#  30
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=(2, 2))) #15

model.add(Dropout(0.25))
model.add(Convolution2D(96, 3, 3, border_mode='same')) #15
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Convolution2D(128, 3, 3))  # 13
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(128, 3, 3,border_mode='same'))  # 13
model.add(MaxPooling2D(pool_size=(2, 2))) #6

model.add(Dropout(0.25))
model.add(Convolution2D(192, 3, 3, border_mode='same')) #6
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, 3)) # 4
model.add(Activation('relu'))
model.add(Convolution2D(512, 3, 3)) # 2
# model.add(MaxPooling2D(pool_size=(2, 2))) #2 

model.add(Dropout(0.25))


model.add(Convolution2D(10, 2))
model.add(Flatten())
model.add(Activation('softmax'))


from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.007 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

~~~

