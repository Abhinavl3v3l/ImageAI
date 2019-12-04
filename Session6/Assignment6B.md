# Networks



~~~python
#6A Model
# model = Sequential()
# model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(32, 32, 3)))# 32
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(Convolution2D(64, 3, 3))#  30
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(Convolution2D(64, 3, 3))#  28
# model.add(Activation('relu'))
# model.add(BatchNormalization())


# model.add(MaxPooling2D(pool_size=(2, 2))) #14

# model.add(Dropout(0.25))
# model.add(Convolution2D(96, 3, 3, border_mode='same')) #14
# model.add(Activation('relu')) 
# model.add(BatchNormalization())
# model.add(Convolution2D(128, 3, 3))  # 13
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Convolution2D(128, 3, 3))  # 11
# model.add(MaxPooling2D(pool_size=(2, 2))) #5

# model.add(Dropout(0.25))
# model.add(Convolution2D(192, 3, 3, border_mode='same')) #5
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Convolution2D(256, 3, 3)) # 3
# model.add(Activation('relu'))

# model.add(Convolution2D(10, 3))
# model.add(Flatten())
# model.add(Activation('softmax'))

#6B Model
# Define the model
# Normal Convolution


# Grouped Convolution (use 3x3, 5x5 only)
# Grouped Convolution (use 3x3 only, one with dilation = 1, and another with dilation = 2)
# You must use all of the 5 above at least once
# Train this new model for 50 epochs.
# Grouped Conv



model = Sequential()
#Normal Convolution 
model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))# 32
model.add(Activation('relu'))
model.add(BatchNormalization())

# Spatially Separable Convolution (Conv2d(x, (3,1)) followed by Conv2D(x,(3,1))
model.add(Convolution2D(64, 3, 1))#  30 x 32
model.add(Convolution2D(64, 1, 3))#  30 x 30
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(10, 1, 1, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) #15
model.add(Dropout(0.1))

# Depthwise Separable Convolution
# model.add(SeparableConv2D(96, (3,3))) #13
model.add(Convolution2D(96, 3, 3, border_mode='same'))   # 15
model.add(Activation('relu')) 
# model.add([Convolution2D(96,(3,3),activation='relu'),Convolution2D(96,(3,3),activation='relu',input=input_shape)])

##############################
nb_filters = 96
kernel_size= {}
kernel_size[0]= [3,3]
kernel_size[1]= [5,5]
# kernel_size[2]= [5,5]
input_shape=(15, 15, 96)

# create seperate model graph for parallel processing with different filter sizes
# apply 'same' padding so that ll produce o/p tensor of same size for concatination
# cancat all paralle output

inp = Input(shape=input_shape)
convs = []
for k_no in range(len(kernel_size)):
    conv = Conv2D(nb_filters, kernel_size[k_no][0], kernel_size[k_no][1],activation='relu',input_shape=input_shape)(inp)
    convs.append(conv)

if len(kernel_size) > 1:
    out = Concatenate(convs,axis = 1)
else:
   out = convs[0]
concat_model = Model(input=inp, output=out)

# add to sequential model
model.add(concat_model) #13
#############################

##############################
nb_filters = 192
kernel_size= {}
kernel_size[0]= [3,3]
kernel_size[1]= [3,3]
input_shape=(13, 13, 96)
dilation_val = [1,2]

# create seperate model graph for parallel processing with different filter sizes
# apply 'same' padding so that ll produce o/p tensor of same size for concatination
# cancat all paralle output

inp = Input(shape=input_shape)
convs = []
for k_no in range(len(kernel_size)):
    conv = Conv2D(nb_filters, kernel_size[k_no][0], kernel_size[k_no][1],border_mode='same',activation='relu',input_shape=input_shape,dilation_rate= dilation_val[k_no])(inp)
    convs.append(conv)

if len(kernel_size) > 1:
    out = Concatenate()(convs)
else:
    out = convs[0]

concat_model = Model(input=inp, output=out)

# add to sequential model
model.add(concat_model)
#############################


model.add(BatchNormalization())
model.add(Convolution2D(96, 3, 3))  # 9
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Convolution2D(10, 1, 1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #5
model.add(Dropout(0.1))

model.add(Convolution2D(192, 3, 3, border_mode='same')) #5
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Convolution2D(192, 3, 3)) # 3
model.add(Activation('relu'))
model.add(BatchNormalization())
# model.add(Convolution2D(192, 3, 3)) # 2
# model.add(MaxPooling2D(pool_size=(2, 2))) #2 

model.add(Convolution2D(10, 3))
model.add(Flatten())
model.add(Activation('softmax'))


# Compile the model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr): return round(0.008 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.008), metrics=['accuracy'])


~~~







----

## 2



~~~

~~~

