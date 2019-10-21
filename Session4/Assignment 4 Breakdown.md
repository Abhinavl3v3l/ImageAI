# Assignment 4 Breakdown



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

### 2 Decided Architecture. 

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





---

Lets try between Starting  Extreme ends and then move towards higher values max = 16

> INCREASING ORDER ONLY

1. 5,8,10, 1x1,12,14,16,  max pool,  16, 18, 20, 1x1
2. 8, 10, 1x1,16, 18, max pool , between 16 and 20
3.  8, 10, 1x1,16, 18, max pool , 16 16 16 
4. 16 and above
5. 

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

