# Assignment 4 Breakdown#

### Though Process.

>  Below is a result of aggressive testing over Assignment 3  without BN, LR  and DO

Over my experiment over vanilla NN architecture(Just conv, max pooling and $1*1$conv) I got pretty much similar result with  greater than 20K and below 20K parameters. 

Result with $ V_{acc} > 20K$ parameter vs $ V_{acc} < 20K$ parameters didn't see much difference plus it was always overfitting results, which is not desired. 

 Which playing with feature and parameters under 15k I noticed skewness in result. The reason being the learning rate. The skewed bowl image shown in session 5 to explain batch normalization concept was visible in result of the network.   Since the bowl was skewed hence the image data was not properly distributed the result went up and down and in loop and took longer epochs to reach in the local minimum.

> After the use of BN, LR  and DO

Only after using Batch Normalization and Dropout  is saw some improvements over the network. 
Learning rate alteration further improved the network to help receive $V_{acc} > 99.4$ . 

All network on longer epochs over-trained themselves. Vanilla network did that from the start. With BN and DO it happened too before I reached validation accuracy of 99.4.  

With Mnist in mind which is a small dataset with one channel and small images with only 10 class to classify. There is a need of few  but rich features. So dedicating over 32 kernel is an over kill.

So before I built an architecture to help reach $V_{acc} \geq  99.4$ , I had to think about  size of the dataset, no of channel used in image, no of kernels to use, size of the image,  and no of classes to identify. As discussed in class a small dataset like **mnist** does not require a lot of kernels, since all it has are edges and gradient and not a lot of textures.  Image size was small and only one grayscale channel to work with. 

I decided to got with three layers before max pooling then 4 layers for high level features of till image size of 5.

|              Network               |
| :--------------------------------: |
|            Convolution             |
|            Convolution             |
| 1 * 1 Convolution -  Rich features |
|             MaxPooling             |
|            Convolution             |
|            Convolution             |
|            Convolution             |
|            Convolution             |

## 1. Vanilla Architecture

~~~python
from keras.layers import Activation
model = Sequential()
# Random large parametere generator. 
model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(28,28,1)))    #26
model.add(Convolution2D(32, 3, 3, activation='relu'))                           #24
model.add(Convolution2D(10, 1, 1, activation='relu'))                           #24
model.add(MaxPooling2D(pool_size=(2, 2)))                                       #12

model.add(Convolution2D(16, 3, 3, activation='relu'))                           #10
model.add(Convolution2D(32, 3, 3, activation='relu'))                           #8
model.add(Convolution2D(32, 3, 3, activation='relu'))                           #6
model.add(Convolution2D(64, 3, 3, activation='relu'))                           #4
model.add(Convolution2D(10, 4))


model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, epochs=20, verbose=1, validation_data=(X_test, Y_test))
~~~

~~~txt
Model: "sequential_26"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_201 (Conv2D)          (None, 26, 26, 8)         80        
_________________________________________________________________
conv2d_202 (Conv2D)          (None, 24, 24, 16)        1168      
_________________________________________________________________
conv2d_203 (Conv2D)          (None, 24, 24, 10)        170       
_________________________________________________________________
max_pooling2d_26 (MaxPooling (None, 12, 12, 10)        0         
_________________________________________________________________
conv2d_204 (Conv2D)          (None, 10, 10, 8)         728       
_________________________________________________________________
conv2d_205 (Conv2D)          (None, 8, 8, 16)          1168      
_________________________________________________________________
conv2d_206 (Conv2D)          (None, 6, 6, 16)          2320      
_________________________________________________________________
conv2d_207 (Conv2D)          (None, 4, 4, 16)          2320      
_________________________________________________________________
conv2d_208 (Conv2D)          (None, 1, 1, 10)          2570      
_________________________________________________________________
flatten_26 (Flatten)         (None, 10)                0         
_________________________________________________________________
activation_26 (Activation)   (None, 10)                0         
=================================================================
Total params: 10,524
Trainable params: 10,524
Non-trainable params: 0

Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 17s 278us/step - loss: 1.1571 - acc: 0.6180 - val_loss: 0.3037 - val_acc: 0.9096
Epoch 2/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.2759 - acc: 0.9163 - val_loss: 0.1969 - val_acc: 0.9400
Epoch 3/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.1911 - acc: 0.9422 - val_loss: 0.2703 - val_acc: 0.9099
Epoch 4/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1521 - acc: 0.9533 - val_loss: 0.1242 - val_acc: 0.9597
Epoch 5/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1290 - acc: 0.9610 - val_loss: 0.1466 - val_acc: 0.9576
Epoch 6/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1130 - acc: 0.9650 - val_loss: 0.0952 - val_acc: 0.9697
Epoch 7/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.1006 - acc: 0.9690 - val_loss: 0.0962 - val_acc: 0.9710
Epoch 8/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.0919 - acc: 0.9716 - val_loss: 0.0928 - val_acc: 0.9724
Epoch 9/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.0846 - acc: 0.9734 - val_loss: 0.0778 - val_acc: 0.9764
Epoch 10/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0799 - acc: 0.9752 - val_loss: 0.0791 - val_acc: 0.9761
Epoch 11/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0744 - acc: 0.9770 - val_loss: 0.1024 - val_acc: 0.9700
Epoch 12/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0711 - acc: 0.9777 - val_loss: 0.0656 - val_acc: 0.9796
Epoch 13/20
60000/60000 [==============================] - 11s 183us/step - loss: 0.0679 - acc: 0.9789 - val_loss: 0.0666 - val_acc: 0.9779
Epoch 14/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0637 - acc: 0.9803 - val_loss: 0.0702 - val_acc: 0.9794
Epoch 15/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0609 - acc: 0.9811 - val_loss: 0.0679 - val_acc: 0.9784
Epoch 16/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0582 - acc: 0.9820 - val_loss: 0.0622 - val_acc: 0.9798
Epoch 17/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0556 - acc: 0.9827 - val_loss: 0.0903 - val_acc: 0.9711
Epoch 18/20
60000/60000 [==============================] - 11s 179us/step - loss: 0.0535 - acc: 0.9832 - val_loss: 0.0598 - val_acc: 0.9823
Epoch 19/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0520 - acc: 0.9838 - val_loss: 0.0612 - val_acc: 0.9810
Epoch 20/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0490 - acc: 0.9845 - val_loss: 0.0725 - val_acc: 0.9776
<keras.callbacks.History at 0x7faa39311c50>
~~~

## 2. Kernel Optimization to fit trainable parameters under 15K

~~~python
# Set up data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)                           
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')                                            
X_test = X_test.astype('float32') 
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)                                  
Y_test = np_utils.to_categorical(y_test, 10)

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))    #26
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #24
model.add(Convolution2D(10, (1, 1), activation='relu'))                           #24

model.add(MaxPooling2D(pool_size=(2, 2)))                                       #12

model.add(Convolution2D(8, (3, 3), activation='relu'))                           #10
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #8
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #6
model.add(Convolution2D(16, (3, 3), activation='relu') )                          #4
model.add(Convolution2D(10, 4))        
model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
~~~

~~~txt
Model: "sequential_26"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_201 (Conv2D)          (None, 26, 26, 8)         80        
_________________________________________________________________
conv2d_202 (Conv2D)          (None, 24, 24, 16)        1168      
_________________________________________________________________
conv2d_203 (Conv2D)          (None, 24, 24, 10)        170       
_________________________________________________________________
max_pooling2d_26 (MaxPooling (None, 12, 12, 10)        0         
_________________________________________________________________
conv2d_204 (Conv2D)          (None, 10, 10, 8)         728       
_________________________________________________________________
conv2d_205 (Conv2D)          (None, 8, 8, 16)          1168      
_________________________________________________________________
conv2d_206 (Conv2D)          (None, 6, 6, 16)          2320      
_________________________________________________________________
conv2d_207 (Conv2D)          (None, 4, 4, 16)          2320      
_________________________________________________________________
conv2d_208 (Conv2D)          (None, 1, 1, 10)          2570      
_________________________________________________________________
flatten_26 (Flatten)         (None, 10)                0         
_________________________________________________________________
activation_26 (Activation)   (None, 10)                0         
=================================================================
Total params: 10,524
Trainable params: 10,524
Non-trainable params: 0

Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 17s 278us/step - loss: 1.1571 - acc: 0.6180 - val_loss: 0.3037 - val_acc: 0.9096
Epoch 2/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.2759 - acc: 0.9163 - val_loss: 0.1969 - val_acc: 0.9400
Epoch 3/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.1911 - acc: 0.9422 - val_loss: 0.2703 - val_acc: 0.9099
Epoch 4/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1521 - acc: 0.9533 - val_loss: 0.1242 - val_acc: 0.9597
Epoch 5/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1290 - acc: 0.9610 - val_loss: 0.1466 - val_acc: 0.9576
Epoch 6/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.1130 - acc: 0.9650 - val_loss: 0.0952 - val_acc: 0.9697
Epoch 7/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.1006 - acc: 0.9690 - val_loss: 0.0962 - val_acc: 0.9710
Epoch 8/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.0919 - acc: 0.9716 - val_loss: 0.0928 - val_acc: 0.9724
Epoch 9/20
60000/60000 [==============================] - 11s 182us/step - loss: 0.0846 - acc: 0.9734 - val_loss: 0.0778 - val_acc: 0.9764
Epoch 10/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0799 - acc: 0.9752 - val_loss: 0.0791 - val_acc: 0.9761
Epoch 11/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0744 - acc: 0.9770 - val_loss: 0.1024 - val_acc: 0.9700
Epoch 12/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0711 - acc: 0.9777 - val_loss: 0.0656 - val_acc: 0.9796
Epoch 13/20
60000/60000 [==============================] - 11s 183us/step - loss: 0.0679 - acc: 0.9789 - val_loss: 0.0666 - val_acc: 0.9779
Epoch 14/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0637 - acc: 0.9803 - val_loss: 0.0702 - val_acc: 0.9794
Epoch 15/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0609 - acc: 0.9811 - val_loss: 0.0679 - val_acc: 0.9784
Epoch 16/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0582 - acc: 0.9820 - val_loss: 0.0622 - val_acc: 0.9798
Epoch 17/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0556 - acc: 0.9827 - val_loss: 0.0903 - val_acc: 0.9711
Epoch 18/20
60000/60000 [==============================] - 11s 179us/step - loss: 0.0535 - acc: 0.9832 - val_loss: 0.0598 - val_acc: 0.9823
Epoch 19/20
60000/60000 [==============================] - 11s 180us/step - loss: 0.0520 - acc: 0.9838 - val_loss: 0.0612 - val_acc: 0.9810
Epoch 20/20
60000/60000 [==============================] - 11s 181us/step - loss: 0.0490 - acc: 0.9845 - val_loss: 0.0725 - val_acc: 0.9776
<keras.callbacks.History at 0x7faa39311c50>
~~~

___

## 3. Go Crazy (Batch Normalization, Drop Out, Learning Rate, Batch Size)

~~~python
# Set up data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)                           
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')                                            
X_test = X_test.astype('float32') 
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)                                  
Y_test = np_utils.to_categorical(y_test, 10)

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(12, 3, 3, activation='relu', input_shape=(28,28,1)))    #26
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(20, 3, 3, activation='relu'))                           #24
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 1, 1, activation='relu'))                           #24
model.add(MaxPooling2D(pool_size=(2, 2)))                                       #12

model.add(Convolution2D(16, 3, 3, activation='relu'))                           #10
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, 3, 3, activation='relu'))                           #8
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, 3, 3, activation='relu'))                           #6
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, 3, 3, activation='relu'))                           #4
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 4))


model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.003 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])
~~~

~~~txt
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_209 (Conv2D)          (None, 26, 26, 12)        120       
_________________________________________________________________
batch_normalization_55 (Batc (None, 26, 26, 12)        48        
_________________________________________________________________
dropout_55 (Dropout)         (None, 26, 26, 12)        0         
_________________________________________________________________
conv2d_210 (Conv2D)          (None, 24, 24, 20)        2180      
_________________________________________________________________
batch_normalization_56 (Batc (None, 24, 24, 20)        80        
_________________________________________________________________
dropout_56 (Dropout)         (None, 24, 24, 20)        0         
_________________________________________________________________
conv2d_211 (Conv2D)          (None, 24, 24, 10)        210       
_________________________________________________________________
max_pooling2d_27 (MaxPooling (None, 12, 12, 10)        0         
_________________________________________________________________
conv2d_212 (Conv2D)          (None, 10, 10, 16)        1456      
_________________________________________________________________
batch_normalization_57 (Batc (None, 10, 10, 16)        64        
_________________________________________________________________
dropout_57 (Dropout)         (None, 10, 10, 16)        0         
_________________________________________________________________
conv2d_213 (Conv2D)          (None, 8, 8, 16)          2320      
_________________________________________________________________
batch_normalization_58 (Batc (None, 8, 8, 16)          64        
_________________________________________________________________
dropout_58 (Dropout)         (None, 8, 8, 16)          0         
_________________________________________________________________
conv2d_214 (Conv2D)          (None, 6, 6, 16)          2320      
_________________________________________________________________
batch_normalization_59 (Batc (None, 6, 6, 16)          64        
_________________________________________________________________
dropout_59 (Dropout)         (None, 6, 6, 16)          0         
_________________________________________________________________
conv2d_215 (Conv2D)          (None, 4, 4, 16)          2320      
_________________________________________________________________
batch_normalization_60 (Batc (None, 4, 4, 16)          64        
_________________________________________________________________
dropout_60 (Dropout)         (None, 4, 4, 16)          0         
_________________________________________________________________
conv2d_216 (Conv2D)          (None, 1, 1, 10)          2570      
_________________________________________________________________
flatten_27 (Flatten)         (None, 10)                0         
_________________________________________________________________
activation_27 (Activation)   (None, 10)                0         
=================================================================
Total params: 13,880
Trainable params: 13,688
Non-trainable params: 192
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 21s 348us/step - loss: 0.2010 - acc: 0.9376 - val_loss: 0.0608 - val_acc: 0.9802
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0618 - acc: 0.9811 - val_loss: 0.0547 - val_acc: 0.9823
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0507 - acc: 0.9843 - val_loss: 0.0382 - val_acc: 0.9887
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0416 - acc: 0.9874 - val_loss: 0.0278 - val_acc: 0.9908
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0377 - acc: 0.9882 - val_loss: 0.0379 - val_acc: 0.9883
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0339 - acc: 0.9890 - val_loss: 0.0249 - val_acc: 0.9931
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0305 - acc: 0.9904 - val_loss: 0.0279 - val_acc: 0.9912
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0288 - acc: 0.9907 - val_loss: 0.0242 - val_acc: 0.9916
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0280 - acc: 0.9911 - val_loss: 0.0206 - val_acc: 0.9936
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0258 - acc: 0.9917 - val_loss: 0.0210 - val_acc: 0.9936
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0247 - acc: 0.9919 - val_loss: 0.0206 - val_acc: 0.9930
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0223 - acc: 0.9927 - val_loss: 0.0211 - val_acc: 0.9942
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0210 - acc: 0.9930 - val_loss: 0.0217 - val_acc: 0.9928
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0208 - acc: 0.9931 - val_loss: 0.0202 - val_acc: 0.9945
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0200 - acc: 0.9937 - val_loss: 0.0199 - val_acc: 0.9941
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 10s 175us/step - loss: 0.0193 - acc: 0.9939 - val_loss: 0.0206 - val_acc: 0.9934
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0182 - acc: 0.9943 - val_loss: 0.0227 - val_acc: 0.9936
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0170 - acc: 0.9945 - val_loss: 0.0223 - val_acc: 0.9928
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0177 - acc: 0.9944 - val_loss: 0.0195 - val_acc: 0.9934
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0175 - acc: 0.9942 - val_loss: 0.0250 - val_acc: 0.9919
<keras.callbacks.History at 0x7faa34bbbf98>
~~~

___

END OF ASSIGNMENT 4 DOCUMENTATION

Below are the architectures used to play around with features and accuracy over Assignment 3 and Assignment 4

---

# Other Architecture and their results

### 1

```python
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))#26  
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #24
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(32, (3, 3), activation='relu'))  #22
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #11

model.add(Convolution2D(16, (3, 3), activation='relu'))  #9
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(32, (3, 3), activation='relu'))  #7
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(20, (3, 3), activation='relu'))  #5
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 5))
model.add(Flatten())
model.add(Activation('softmax'))
```

~~~text
Total params: 20,010
Trainable params: 19,742
Non-trainable params: 268
~~~

~~~txt
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
  """Entry point for launching an IPython kernel.
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 25s 414us/step - loss: 0.2640 - acc: 0.9169 - val_loss: 0.0545 - val_acc: 0.9819
Epoch 2/20
60000/60000 [==============================] - 22s 373us/step - loss: 0.0747 - acc: 0.9768 - val_loss: 0.0418 - val_acc: 0.9853
Epoch 3/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0587 - acc: 0.9817 - val_loss: 0.0356 - val_acc: 0.9889
Epoch 4/20
60000/60000 [==============================] - 22s 372us/step - loss: 0.0481 - acc: 0.9853 - val_loss: 0.0253 - val_acc: 0.9922
Epoch 5/20
60000/60000 [==============================] - 22s 374us/step - loss: 0.0441 - acc: 0.9867 - val_loss: 0.0305 - val_acc: 0.9911
Epoch 6/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0394 - acc: 0.9875 - val_loss: 0.0280 - val_acc: 0.9916
Epoch 7/20
60000/60000 [==============================] - 22s 369us/step - loss: 0.0374 - acc: 0.9881 - val_loss: 0.0304 - val_acc: 0.9901
Epoch 8/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0364 - acc: 0.9885 - val_loss: 0.0292 - val_acc: 0.9908
Epoch 9/20
60000/60000 [==============================] - 22s 368us/step - loss: 0.0330 - acc: 0.9892 - val_loss: 0.0241 - val_acc: 0.9922
Epoch 10/20
60000/60000 [==============================] - 22s 371us/step - loss: 0.0298 - acc: 0.9902 - val_loss: 0.0341 - val_acc: 0.9895
Epoch 11/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0306 - acc: 0.9904 - val_loss: 0.0217 - val_acc: 0.9931
Epoch 12/20
60000/60000 [==============================] - 23s 380us/step - loss: 0.0276 - acc: 0.9912 - val_loss: 0.0239 - val_acc: 0.9915
Epoch 13/20
60000/60000 [==============================] - 23s 380us/step - loss: 0.0271 - acc: 0.9909 - val_loss: 0.0235 - val_acc: 0.9925
Epoch 14/20
60000/60000 [==============================] - 23s 390us/step - loss: 0.0253 - acc: 0.9916 - val_loss: 0.0254 - val_acc: 0.9921
Epoch 15/20
60000/60000 [==============================] - 23s 386us/step - loss: 0.0266 - acc: 0.9915 - val_loss: 0.0209 - val_acc: 0.9941
Epoch 16/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0236 - acc: 0.9924 - val_loss: 0.0195 - val_acc: 0.9935
Epoch 17/20
60000/60000 [==============================] - 23s 388us/step - loss: 0.0216 - acc: 0.9929 - val_loss: 0.0257 - val_acc: 0.9916
Epoch 18/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0226 - acc: 0.9925 - val_loss: 0.0227 - val_acc: 0.9924
Epoch 19/20
60000/60000 [==============================] - 23s 379us/step - loss: 0.0220 - acc: 0.9927 - val_loss: 0.0244 - val_acc: 0.9927
Epoch 20/20
60000/60000 [==============================] - 23s 381us/step - loss: 0.0206 - acc: 0.9934 - val_loss: 0.0213 - val_acc: 0.9943
<keras.callbacks.History at 0x7f2557612630>
~~~

---

### 2 

~~~python
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(10, (3, 3), activation='relu', input_shape=(28,28,1)))#26  
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #24
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))

model.add(Convolution2D(20, (3, 3), activation='relu'))  #22
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) #11

model.add(Convolution2D(10, (3, 3), activation='relu'))  #9
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, (3, 3), activation='relu'))  #7
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #5
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #3
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 3))
model.add(Flatten())
model.add(Activation('softmax'))		
~~~

~~~txt
Total params: 12,664
Trainable params: 12,436
Non-trainable params: 228
~~~



Not worth it.

---

### 3

~~~python
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))#26  
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, activation='relu'))#22

model.add(Convolution2D(16, (3, 3), activation='relu'))  #24
model.add(BatchNormalization())
model.add(Dropout(0.1))



model.add(Convolution2D(16, (3, 3), activation='relu'))  #22
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) #11


model.add(Convolution2D(16, (3, 3), activation='relu'))  #9
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, (3, 3), activation='relu'))  #7
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, (3, 3), activation='relu'))  #5
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 5))
model.add(Flatten())
model.add(Activation('softmax'))	
~~~



~~~
Total params: 15,958
Trainable params: 15,650
Non-trainable params: 308
~~~

~~~txt
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 26s 438us/step - loss: 0.2816 - acc: 0.9117 - val_loss: 0.0564 - val_acc: 0.9816
Epoch 2/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0755 - acc: 0.9759 - val_loss: 0.0378 - val_acc: 0.9867
Epoch 3/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0588 - acc: 0.9814 - val_loss: 0.0565 - val_acc: 0.9832
Epoch 4/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0498 - acc: 0.9846 - val_loss: 0.0305 - val_acc: 0.9904
Epoch 5/20
60000/60000 [==============================] - 23s 383us/step - loss: 0.0463 - acc: 0.9854 - val_loss: 0.0336 - val_acc: 0.9893
Epoch 6/20
60000/60000 [==============================] - 23s 381us/step - loss: 0.0412 - acc: 0.9872 - val_loss: 0.0250 - val_acc: 0.9926
Epoch 7/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0378 - acc: 0.9882 - val_loss: 0.0239 - val_acc: 0.9927
Epoch 8/20
60000/60000 [==============================] - 23s 381us/step - loss: 0.0363 - acc: 0.9885 - val_loss: 0.0287 - val_acc: 0.9911
Epoch 9/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0341 - acc: 0.9893 - val_loss: 0.0224 - val_acc: 0.9938
Epoch 10/20
60000/60000 [==============================] - 23s 383us/step - loss: 0.0341 - acc: 0.9894 - val_loss: 0.0265 - val_acc: 0.9916
Epoch 11/20
60000/60000 [==============================] - 23s 381us/step - loss: 0.0315 - acc: 0.9897 - val_loss: 0.0228 - val_acc: 0.9923
Epoch 12/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0292 - acc: 0.9906 - val_loss: 0.0243 - val_acc: 0.9924
Epoch 13/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0283 - acc: 0.9910 - val_loss: 0.0208 - val_acc: 0.9931
Epoch 14/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0274 - acc: 0.9912 - val_loss: 0.0316 - val_acc: 0.9909
Epoch 15/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0270 - acc: 0.9911 - val_loss: 0.0276 - val_acc: 0.9917
Epoch 16/20
60000/60000 [==============================] - 23s 383us/step - loss: 0.0253 - acc: 0.9916 - val_loss: 0.0217 - val_acc: 0.9930
Epoch 17/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0255 - acc: 0.9915 - val_loss: 0.0223 - val_acc: 0.9929
Epoch 18/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0236 - acc: 0.9923 - val_loss: 0.0213 - val_acc: 0.9940
Epoch 19/20
60000/60000 [==============================] - 23s 383us/step - loss: 0.0225 - acc: 0.9925 - val_loss: 0.0220 - val_acc: 0.9929
Epoch 20/20
60000/60000 [==============================] - 23s 384us/step - loss: 0.0232 - acc: 0.9926 - val_loss: 0.0221 - val_acc: 0.9939
<keras.callbacks.History at 0x7fce5b5ee6d8>
~~~





---

### 4

~~~txt
# Set up data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)                           
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_train = X_train.astype('float32')                                            
X_test = X_test.astype('float32') 
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)                                  
Y_test = np_utils.to_categorical(y_test, 10)

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))    #26
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #24
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(10, (1, 1), activation='relu'))                           #24

model.add(MaxPooling2D(pool_size=(2, 2)))                                       #12

model.add(Convolution2D(8, (3, 3), activation='relu'))                           #10
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #8
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, (3, 3), activation='relu'))                           #6
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Convolution2D(16, (3, 3), activation='relu') )                          #4
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 4))        
model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
~~~

> Was able to achieve 99.42 but after overfitting, but result was not overfitted.

___

