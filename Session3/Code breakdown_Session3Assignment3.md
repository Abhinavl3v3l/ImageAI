# Code breakdown





Imnist Dataset



| Input Size | Kernel Size | Output Size | Parameters  |
| ---------- | ----------- | ----------- | ----------- |
| 28*28\*1   | 3*3\*1\*8   | 26*26\*6    | 72 + 8 = 80 |
| 26*26\*8   | 3*3\*8\*16  | 24*24\*16   | +16 =       |
|            |             |             |             |





Since MNIST has input image of 28x28.  We do not need  very large number of kernel, as there are no textures and there is only one input channel initially.

Way to reduce parameter is to decrease number of kernel from initial value  of 32 to maybe 8. 
Hence making 3 Convolution layer 1 Max Pool and then 2-3 Max pool layer to reduce the size to 7 or 9 pixels.





> 992

```python
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))#26  

model.add(Convolution2D(16, (3, 3), activation='relu'))  #24
model.add(Convolution2D(32, (3, 3), activation='relu'))  #22


model.add(MaxPooling2D(pool_size=(2, 2))) #11

model.add(Convolution2D(16, (3, 3), activation='relu'))  #9
model.add(Convolution2D(32, (3, 3), activation='relu'))  #7

model.add(Convolution2D(32, 1, activation='relu'))
model.add(Convolution2D(10, 7))

model.add(Flatten())
model.add(Activation('softmax')) 

```

---



>9917

~~~
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))#26  


model.add(MaxPooling2D(pool_size=(2, 2))) #13

model.add(Convolution2D(8, (3, 3), activation='relu'))  #11
model.add(Convolution2D(16, (3, 3), activation='relu'))  #9

model.add(Convolution2D(16, 1, activation='relu'))

model.add(Convolution2D(8, (3, 3), activation='relu'))  #7
model.add(Convolution2D(16, (3, 3), activation='relu'))  #5

model.add(Convolution2D(32, 1, activation='relu'))
model.add(Convolution2D(10, 5))

model.add(Flatten())
model.add(Activation('softmax'))
~~~

---



>9917 100epochs

~~~
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))#26 
model.add(Convolution2D(8, (3, 3), activation='relu'))  #24
model.add(Convolution2D(16, (3, 3), activation='relu'))  #22
model.add(Convolution2D(16, (3, 3), activation='relu'))  #20
model.add(Convolution2D(32, (3, 3), activation='relu'))  #18

model.add(MaxPooling2D(pool_size=(2, 2))) #9

model.add(Convolution2D(8, (3, 3), activation='relu'))  #7
model.add(Convolution2D(16, (3, 3), activation='relu'))  #5

model.add(Convolution2D(20, 1, activation='relu'))
model.add(Convolution2D(10, 5))

model.add(Flatten())
model.add(Activation('softmax'))
~~~

---

> 9917

~~~
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(28,28,1)))#26 
model.add(Convolution2D(32, (3, 3), activation='relu'))  #24


model.add(MaxPooling2D(pool_size=(2, 2))) #12

model.add(Convolution2D(16, (3, 3), activation='relu'))  #10
model.add(Convolution2D(32, (3, 3), activation='relu'))  #8

model.add(Convolution2D(10, 1, activation='relu'))
model.add(Convolution2D(10, 8))

model.add(Flatten())
model.add(Activation('softmax'))
~~~

---

All layers give same as divided as above



---



> 9928

~~~
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(8, (3, 3), activation='relu', input_shape=(28,28,1)))#26 
model.add(Convolution2D(16, (3, 3), activation='relu'))  #24
model.add(Convolution2D(16, (3, 3), activation='relu'))  #22
model.add(MaxPooling2D(pool_size=(2, 2))) #11
model.add(Convolution2D(8, (3, 3), activation='relu'))  #9
model.add(Convolution2D(16, (3, 3), activation='relu'))  #7
model.add(Convolution2D(20, 1, activation='relu'))
model.add(Convolution2D(30, (3, 3), activation='relu'))  #5

model.add(Convolution2D(10, 5))

model.add(Flatten())
model.add(Activation('softmax'))
~~~

---



### 9917  - Essentia

~~~
# Building the layers for maximum accuracy
from keras.layers import Activation
model = Sequential()


model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))  # Input Image size 28x28x1, 3x3 convolution with 32 channels,     GRF = 3, Parameters = 3x3x32x1  + 32        = 320           Output Image = 26x26x32

model.add(Convolution2D(16, 1, activation='relu'))                              

model.add(MaxPooling2D(pool_size=(2, 2))) #13

model.add(Convolution2D(8, (3, 3), activation='relu'))  #11
model.add(Convolution2D(16, (3, 3), activation='relu'))  #9
model.add(Convolution2D(32, (3, 3), activation='relu'))  #7
model.add(Convolution2D(32, (3, 3), activation='relu'))  #5

model.add(Convolution2D(10, 5))

model.add(Flatten())
model.add(Activation('softmax'))
~~~

