- There is no wiki page as the same wiki content has been included in jupyter notebook (scroll below)
- [My Jupyter Notebook](https://github.com/Ruthvicp/CS5590_PyDL/blob/master/Module2/In_Class_Exercise/ICE4/Source/Image_Classification_CNN.ipynb)
- [Youtube demo](https://youtu.be/POSK5Jupfik)
- Also the source code can be found in the form of .py file [here](https://github.com/Ruthvicp/CS5590_PyDL/tree/master/Module2/In_Class_Exercise/ICE4/Source)


## Image Classification with CNN, Dataset - MNIST
### author:ruthvicp
### date : April 19, 2019


```python
# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
```


```python
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170500096/170498071 [==============================] - 9s 0us/step
    


```python
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

```


```python
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

```


```python
# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    


```python
import keras
tbCallBack= keras.callbacks.TensorBoard(log_dir='./Graph', write_images=True)

```


```python
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32, 32, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 32, 32, 32)        9248      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 32, 16, 16)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 8192)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 512)               4194816   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 4,210,090
    Trainable params: 4,210,090
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, callbacks=[tbCallBack])
```

    Train on 50000 samples, validate on 10000 samples
    Epoch 1/25
    50000/50000 [==============================] - 10s 204us/step - loss: 0.8147 - acc: 0.7118 - val_loss: 0.9516 - val_acc: 0.6760
    Epoch 2/25
    50000/50000 [==============================] - 10s 203us/step - loss: 0.5716 - acc: 0.7994 - val_loss: 1.0123 - val_acc: 0.6788
    Epoch 3/25
    50000/50000 [==============================] - 10s 200us/step - loss: 0.4275 - acc: 0.8483 - val_loss: 1.0286 - val_acc: 0.6995
    Epoch 4/25
    50000/50000 [==============================] - 10s 204us/step - loss: 0.3353 - acc: 0.8827 - val_loss: 1.0235 - val_acc: 0.7011
    Epoch 5/25
    50000/50000 [==============================] - 10s 210us/step - loss: 0.2750 - acc: 0.9026 - val_loss: 1.0521 - val_acc: 0.7101
    Epoch 6/25
    50000/50000 [==============================] - 10s 201us/step - loss: 0.2326 - acc: 0.9174 - val_loss: 1.0931 - val_acc: 0.7118
    Epoch 7/25
    50000/50000 [==============================] - 10s 199us/step - loss: 0.1989 - acc: 0.9307 - val_loss: 1.1702 - val_acc: 0.7102
    Epoch 8/25
    50000/50000 [==============================] - 10s 208us/step - loss: 0.1761 - acc: 0.9392 - val_loss: 1.1461 - val_acc: 0.7157
    Epoch 9/25
    50000/50000 [==============================] - 10s 199us/step - loss: 0.1592 - acc: 0.9451 - val_loss: 1.1664 - val_acc: 0.7149
    Epoch 10/25
    50000/50000 [==============================] - 10s 198us/step - loss: 0.1450 - acc: 0.9494 - val_loss: 1.1945 - val_acc: 0.7189
    Epoch 11/25
    50000/50000 [==============================] - 10s 199us/step - loss: 0.1327 - acc: 0.9545 - val_loss: 1.2169 - val_acc: 0.7178
    Epoch 12/25
    50000/50000 [==============================] - 10s 205us/step - loss: 0.1245 - acc: 0.9569 - val_loss: 1.2150 - val_acc: 0.7186
    Epoch 13/25
    50000/50000 [==============================] - 10s 209us/step - loss: 0.1121 - acc: 0.9617 - val_loss: 1.2655 - val_acc: 0.7133
    Epoch 14/25
    50000/50000 [==============================] - 10s 201us/step - loss: 0.1071 - acc: 0.9627 - val_loss: 1.2536 - val_acc: 0.7195
    Epoch 15/25
    50000/50000 [==============================] - 10s 200us/step - loss: 0.1003 - acc: 0.9658 - val_loss: 1.2938 - val_acc: 0.7195
    Epoch 16/25
    50000/50000 [==============================] - 10s 200us/step - loss: 0.0972 - acc: 0.9667 - val_loss: 1.2881 - val_acc: 0.7178
    Epoch 17/25
    50000/50000 [==============================] - 10s 200us/step - loss: 0.0897 - acc: 0.9695 - val_loss: 1.3073 - val_acc: 0.7202
    Epoch 18/25
    50000/50000 [==============================] - 10s 200us/step - loss: 0.0851 - acc: 0.9711 - val_loss: 1.3541 - val_acc: 0.7185
    Epoch 19/25
    50000/50000 [==============================] - 10s 200us/step - loss: 0.0837 - acc: 0.9718 - val_loss: 1.3143 - val_acc: 0.7215
    Epoch 20/25
    50000/50000 [==============================] - 10s 204us/step - loss: 0.0804 - acc: 0.9719 - val_loss: 1.3365 - val_acc: 0.7195
    Epoch 21/25
    50000/50000 [==============================] - 10s 207us/step - loss: 0.0783 - acc: 0.9736 - val_loss: 1.3370 - val_acc: 0.7217
    Epoch 22/25
    50000/50000 [==============================] - 10s 200us/step - loss: 0.0752 - acc: 0.9745 - val_loss: 1.3554 - val_acc: 0.7194
    Epoch 23/25
    50000/50000 [==============================] - 10s 200us/step - loss: 0.0752 - acc: 0.9745 - val_loss: 1.3441 - val_acc: 0.7194
    Epoch 24/25
    50000/50000 [==============================] - 10s 198us/step - loss: 0.0704 - acc: 0.9766 - val_loss: 1.3562 - val_acc: 0.7175
    Epoch 25/25
    50000/50000 [==============================] - 10s 198us/step - loss: 0.0700 - acc: 0.9761 - val_loss: 1.3528 - val_acc: 0.7212
    




    <keras.callbacks.History at 0x7fc84be9d518>




```python
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("loss: %.2f%%" % (scores[0]*100))
```

    Accuracy: 72.12%
    
## Now let's modify the model according to the ICP - add a CNN layer with below layers

- Convolutional input layer, 32 feature maps with a size of 3×3 and a rectifier activation function.
- Dropout layer at 20%.
- Convolutional layer, 32 feature maps with a size of 3×3 and a rectifier activation function.
- Max Pool layer with size 2×2.Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function.
- Dropout layer at 20%.
- Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function.
- Max Pool layer with size 2×2.
- Convolutional layer, 128feature maps with a size of 3×3 and a rectifier activation function.
- Dropout layer at 20%. 
- Convolutional layer,128 feature maps with a size of 3×3 and a rectifier activation function.
- Max Pool layer with size 2×2.
- Flatten layer. 
- Dropout layer at 20%.
- Fully connected layer with 1024units and a rectifier activation function.
- Dropout layer at 20%.Fully connected layer with 512units and a rectifier activation function.
- Dropoutlayer at 20%.
- Fully connected output layer with 10 units and a softmax activation function

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# flattening the matrix into vector form
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
```


```python
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_3 (Conv2D)            (None, 32, 32, 32)        896       
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 32, 32, 32)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 32, 32, 32)        9248      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 32, 16, 16)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 64, 16, 16)        18496     
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 64, 16, 16)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 64, 16, 16)        36928     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 64, 8, 8)          0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 128, 8, 8)         73856     
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 128, 8, 8)         0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 128, 8, 8)         147584    
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 128, 4, 4)         0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 2048)              0         
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 2048)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1024)              2098176   
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 512)               524800    
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 2,915,114
    Trainable params: 2,915,114
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32,\
          callbacks=[tbCallBack])
```

    Train on 50000 samples, validate on 10000 samples
    Epoch 1/25
    50000/50000 [==============================] - 13s 259us/step - loss: 1.9103 - acc: 0.2932 - val_loss: 1.5403 - val_acc: 0.4327
    Epoch 2/25
    50000/50000 [==============================] - 13s 258us/step - loss: 1.4194 - acc: 0.4822 - val_loss: 1.3054 - val_acc: 0.5204
    Epoch 3/25
    50000/50000 [==============================] - 13s 254us/step - loss: 1.1792 - acc: 0.5765 - val_loss: 1.0766 - val_acc: 0.6088
    Epoch 4/25
    50000/50000 [==============================] - 13s 251us/step - loss: 1.0216 - acc: 0.6365 - val_loss: 1.0025 - val_acc: 0.6462
    Epoch 5/25
    50000/50000 [==============================] - 13s 251us/step - loss: 0.9052 - acc: 0.6800 - val_loss: 0.8534 - val_acc: 0.6982
    Epoch 6/25
    50000/50000 [==============================] - 13s 259us/step - loss: 0.8270 - acc: 0.7078 - val_loss: 0.7984 - val_acc: 0.7173
    Epoch 7/25
    50000/50000 [==============================] - 12s 250us/step - loss: 0.7677 - acc: 0.7302 - val_loss: 0.7867 - val_acc: 0.7236
    Epoch 8/25
    50000/50000 [==============================] - 13s 253us/step - loss: 0.7056 - acc: 0.7504 - val_loss: 0.7093 - val_acc: 0.7551
    Epoch 9/25
    50000/50000 [==============================] - 13s 258us/step - loss: 0.6690 - acc: 0.7624 - val_loss: 0.7054 - val_acc: 0.7529
    Epoch 10/25
    50000/50000 [==============================] - 13s 266us/step - loss: 0.6346 - acc: 0.7760 - val_loss: 0.6905 - val_acc: 0.7592
    Epoch 11/25
    50000/50000 [==============================] - 13s 255us/step - loss: 0.5983 - acc: 0.7888 - val_loss: 0.6559 - val_acc: 0.7731
    Epoch 12/25
    50000/50000 [==============================] - 13s 256us/step - loss: 0.5668 - acc: 0.7981 - val_loss: 0.6628 - val_acc: 0.7716
    Epoch 13/25
    50000/50000 [==============================] - 13s 255us/step - loss: 0.5394 - acc: 0.8091 - val_loss: 0.6339 - val_acc: 0.7799
    Epoch 14/25
    50000/50000 [==============================] - 13s 255us/step - loss: 0.5180 - acc: 0.8156 - val_loss: 0.6206 - val_acc: 0.7821
    Epoch 15/25
    50000/50000 [==============================] - 13s 268us/step - loss: 0.4963 - acc: 0.8229 - val_loss: 0.6315 - val_acc: 0.7847
    Epoch 16/25
    50000/50000 [==============================] - 13s 256us/step - loss: 0.4744 - acc: 0.8313 - val_loss: 0.6150 - val_acc: 0.7884
    Epoch 17/25
    50000/50000 [==============================] - 13s 255us/step - loss: 0.4577 - acc: 0.8362 - val_loss: 0.6059 - val_acc: 0.7887
    Epoch 18/25
    50000/50000 [==============================] - 13s 255us/step - loss: 0.4356 - acc: 0.8464 - val_loss: 0.6236 - val_acc: 0.7887
    Epoch 19/25
    50000/50000 [==============================] - 13s 256us/step - loss: 0.4225 - acc: 0.8504 - val_loss: 0.6133 - val_acc: 0.7936
    Epoch 20/25
    50000/50000 [==============================] - 13s 255us/step - loss: 0.4078 - acc: 0.8541 - val_loss: 0.5980 - val_acc: 0.7987
    Epoch 21/25
    50000/50000 [==============================] - 13s 267us/step - loss: 0.3944 - acc: 0.8603 - val_loss: 0.6095 - val_acc: 0.7934
    Epoch 22/25
    50000/50000 [==============================] - 13s 257us/step - loss: 0.3754 - acc: 0.8675 - val_loss: 0.6003 - val_acc: 0.7966
    Epoch 23/25
    50000/50000 [==============================] - 13s 269us/step - loss: 0.3641 - acc: 0.8690 - val_loss: 0.6124 - val_acc: 0.7969
    Epoch 24/25
    50000/50000 [==============================] - 13s 260us/step - loss: 0.3535 - acc: 0.8742 - val_loss: 0.6173 - val_acc: 0.7959
    Epoch 25/25
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3428 - acc: 0.8763 - val_loss: 0.6012 - val_acc: 0.8023
    




    <keras.callbacks.History at 0x7fc84a498128>




```python
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Loss: %.2f%%" % (scores[0]*100))
```

    Accuracy: 80.23%
    Loss: 60.12%
    


```python
print(X_test.shape)
print(y_test.shape)
```

    (10000, 3, 32, 32)
    (10000, 10)
    


```python
import pandas as pd
prediction = pd.DataFrame()
imageid = []
for i in range(len(X_test)):
    i = i + 1
    imageid.append(i)
prediction["ImageId"] = imageid
prediction["Label"] = model.predict_classes(X_test, verbose=0)
print(prediction.head())

```

       ImageId  Label
    0        1      3
    1        2      8
    2        3      8
    3        4      0
    4        5      6
    


```python
import numpy as np  
a  = np.array(y_test[0:5])
print('Actual labels for first five images: {0}'.format(np.argmax(a, axis=1)))
```

    Actual labels for first five images: [3 8 8 0 6]
    

### Accuracy Plot from Tensorboard
![Accuracy_plot](https://github.com/Ruthvicp/CS5590_PyDL/raw/master/Module2/In_Class_Exercise/ICE4/Documentation/1.JPG)

### Loss plot from tensorboard

![Loss_plot](https://github.com/Ruthvicp/CS5590_PyDL/raw/master/Module2/In_Class_Exercise/ICE4/Documentation/2.JPG)

### Remarks
- The second model has better accuracy (82%) compared to previous model (72%)
- The trade off here is, the second model takes a bit long to train
- Predicted labels for first 5 images = [3 8 8 0 6]
- Actual labels for first 5 images = [3 8 8 0 6]
