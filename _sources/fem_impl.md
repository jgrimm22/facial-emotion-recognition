# Implementation

Beispielimplementation FER mithilfe eines CNN
Basierend auf https://github.com/gitshanks/fer2013


```python
import sys, os
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

    Using TensorFlow backend.


### Daten laden

In diesem Abschnitt werden die Daten und Labels des FER2013 Datensatz aus einer .csv Datei geladen und vorverarbeitet.


```python
data = pd.read_csv('data/fer2013.csv')
```


```python
width, height = 48, 48

datapoints = data['pixels'].tolist()

#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

#getting labels for training
y = pd.get_dummies(data['emotion']).values

#data normalization
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

print("Preprocessing Done")
print("Number of Features: "+str(len(X[0])))
print("Number of Labels: "+ str(len(y[0])))
print("Number of examples in dataset:"+str(len(X)))
```

    Preprocessing Done
    Number of Features: 48
    Number of Labels: 7
    Number of examples in dataset:35887


### Daten aufteilen

Hier werden die Daten in Train, Test und Validierungsdaten aufgeteilt und die Testdaten für später gespeichert.


```python
#splitting into training, validation and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

#saving the test samples to be used later
np.save('modXtest', X_test)
np.save('modytest', y_test)
```

### Modell definieren

![Eigenes Modell](./images/ownModel.jpg)

```python
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48
```


```python
#desinging the CNN
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

#model.summary()
```

    WARNING:tensorflow:From /home/jh275/anaconda3/envs/matterport/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    WARNING:tensorflow:From /home/jh275/anaconda3/envs/matterport/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    


### Modell trainieren


```python
#Compliling the model with adam optimixer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
```


```python
#training the model
model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_valid), np.array(y_valid)),
          shuffle=True)
```

    WARNING:tensorflow:From /home/jh275/anaconda3/envs/matterport/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    Train on 29068 samples, validate on 3230 samples
    Epoch 1/100
    29068/29068 [==============================] - 17s 573us/step - loss: 1.9859 - accuracy: 0.2133 - val_loss: 1.8271 - val_accuracy: 0.2594
    Epoch 2/100
    29068/29068 [==============================] - 15s 499us/step - loss: 1.8312 - accuracy: 0.2459 - val_loss: 1.8081 - val_accuracy: 0.2616
    Epoch 3/100
    29068/29068 [==============================] - 15s 503us/step - loss: 1.7654 - accuracy: 0.2820 - val_loss: 1.7208 - val_accuracy: 0.3022
    Epoch 4/100
    29068/29068 [==============================] - 15s 533us/step - loss: 1.6388 - accuracy: 0.3501 - val_loss: 1.5385 - val_accuracy: 0.4025
    Epoch 5/100
    29068/29068 [==============================] - 16s 535us/step - loss: 1.5315 - accuracy: 0.4069 - val_loss: 1.4257 - val_accuracy: 0.4384
    Epoch 6/100
    29068/29068 [==============================] - 15s 514us/step - loss: 1.4654 - accuracy: 0.4344 - val_loss: 1.3449 - val_accuracy: 0.4789
    Epoch 7/100
    29068/29068 [==============================] - 16s 540us/step - loss: 1.4075 - accuracy: 0.4570 - val_loss: 1.3001 - val_accuracy: 0.4941
    Epoch 8/100
    29068/29068 [==============================] - 16s 545us/step - loss: 1.3739 - accuracy: 0.4753 - val_loss: 1.2503 - val_accuracy: 0.5235
    Epoch 9/100
    29068/29068 [==============================] - 16s 549us/step - loss: 1.3343 - accuracy: 0.4960 - val_loss: 1.2543 - val_accuracy: 0.5124
    Epoch 10/100
    29068/29068 [==============================] - 15s 530us/step - loss: 1.3042 - accuracy: 0.5098 - val_loss: 1.2365 - val_accuracy: 0.5300
    Epoch 11/100
    29068/29068 [==============================] - 16s 556us/step - loss: 1.2820 - accuracy: 0.5148 - val_loss: 1.1802 - val_accuracy: 0.5464
    Epoch 12/100
    29068/29068 [==============================] - 16s 555us/step - loss: 1.2572 - accuracy: 0.5282 - val_loss: 1.1446 - val_accuracy: 0.5650
    Epoch 13/100
    29068/29068 [==============================] - 16s 549us/step - loss: 1.2323 - accuracy: 0.5374 - val_loss: 1.1449 - val_accuracy: 0.5669
    Epoch 14/100
    29068/29068 [==============================] - 16s 544us/step - loss: 1.2102 - accuracy: 0.5485 - val_loss: 1.1264 - val_accuracy: 0.5796
    Epoch 15/100
    29068/29068 [==============================] - 16s 543us/step - loss: 1.1913 - accuracy: 0.5559 - val_loss: 1.1341 - val_accuracy: 0.5728
    Epoch 16/100
    29068/29068 [==============================] - 16s 544us/step - loss: 1.1655 - accuracy: 0.5669 - val_loss: 1.1105 - val_accuracy: 0.5997
    Epoch 17/100
    29068/29068 [==============================] - 16s 555us/step - loss: 1.1414 - accuracy: 0.5831 - val_loss: 1.1002 - val_accuracy: 0.5954ss: 1
    Epoch 18/100
    29068/29068 [==============================] - 16s 548us/step - loss: 1.1247 - accuracy: 0.5903 - val_loss: 1.0881 - val_accuracy: 0.5957
    Epoch 19/100
    29068/29068 [==============================] - 16s 555us/step - loss: 1.0992 - accuracy: 0.6018 - val_loss: 1.0756 - val_accuracy: 0.6102
    Epoch 20/100
    29068/29068 [==============================] - 16s 536us/step - loss: 1.0767 - accuracy: 0.6071 - val_loss: 1.0814 - val_accuracy: 0.6015
    Epoch 21/100
    29068/29068 [==============================] - 16s 553us/step - loss: 1.0576 - accuracy: 0.6166 - val_loss: 1.0574 - val_accuracy: 0.6158
    Epoch 22/100
    29068/29068 [==============================] - 16s 553us/step - loss: 1.0281 - accuracy: 0.6246 - val_loss: 1.0496 - val_accuracy: 0.6217
    Epoch 23/100
    29068/29068 [==============================] - 16s 555us/step - loss: 1.0118 - accuracy: 0.6371 - val_loss: 1.0500 - val_accuracy: 0.6251
    Epoch 24/100
    29068/29068 [==============================] - 16s 554us/step - loss: 1.0012 - accuracy: 0.6383 - val_loss: 1.0352 - val_accuracy: 0.6192
    Epoch 25/100
    29068/29068 [==============================] - 16s 550us/step - loss: 0.9743 - accuracy: 0.6476 - val_loss: 1.0145 - val_accuracy: 0.6334
    Epoch 26/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.9594 - accuracy: 0.6558 - val_loss: 1.0335 - val_accuracy: 0.6381
    Epoch 27/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.9404 - accuracy: 0.6605 - val_loss: 1.0604 - val_accuracy: 0.6331
    Epoch 28/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.9249 - accuracy: 0.6710 - val_loss: 1.0147 - val_accuracy: 0.6402
    Epoch 29/100
    29068/29068 [==============================] - 15s 533us/step - loss: 0.9018 - accuracy: 0.6756 - val_loss: 1.0157 - val_accuracy: 0.6412
    Epoch 30/100
    29068/29068 [==============================] - 16s 557us/step - loss: 0.8818 - accuracy: 0.6855 - val_loss: 0.9909 - val_accuracy: 0.6474
    Epoch 31/100
    29068/29068 [==============================] - 16s 548us/step - loss: 0.8662 - accuracy: 0.6861 - val_loss: 1.0033 - val_accuracy: 0.6483
    Epoch 32/100
    29068/29068 [==============================] - 16s 557us/step - loss: 0.8523 - accuracy: 0.6972 - val_loss: 1.0042 - val_accuracy: 0.6458
    Epoch 33/100
    29068/29068 [==============================] - 16s 547us/step - loss: 0.8299 - accuracy: 0.7029 - val_loss: 0.9896 - val_accuracy: 0.6415uracy: 0.
    Epoch 34/100
    29068/29068 [==============================] - 16s 552us/step - loss: 0.8214 - accuracy: 0.7072 - val_loss: 0.9913 - val_accuracy: 0.6573
    Epoch 35/100
    29068/29068 [==============================] - 16s 547us/step - loss: 0.8049 - accuracy: 0.7146 - val_loss: 0.9850 - val_accuracy: 0.6526
    Epoch 36/100
    29068/29068 [==============================] - 16s 553us/step - loss: 0.7807 - accuracy: 0.7196 - val_loss: 1.0330 - val_accuracy: 0.6567
    Epoch 37/100
    29068/29068 [==============================] - 16s 553us/step - loss: 0.7698 - accuracy: 0.7273 - val_loss: 1.0231 - val_accuracy: 0.6502
    Epoch 38/100
    29068/29068 [==============================] - 16s 556us/step - loss: 0.7540 - accuracy: 0.7335 - val_loss: 1.0408 - val_accuracy: 0.6492
    Epoch 39/100
    29068/29068 [==============================] - 16s 553us/step - loss: 0.7369 - accuracy: 0.7379 - val_loss: 1.0609 - val_accuracy: 0.6523
    Epoch 40/100
    29068/29068 [==============================] - 16s 551us/step - loss: 0.7285 - accuracy: 0.7455 - val_loss: 1.0298 - val_accuracy: 0.6588
    Epoch 41/100
    29068/29068 [==============================] - 16s 552us/step - loss: 0.7078 - accuracy: 0.7501 - val_loss: 1.0534 - val_accuracy: 0.6576
    Epoch 42/100
    29068/29068 [==============================] - 16s 554us/step - loss: 0.6940 - accuracy: 0.7556 - val_loss: 1.0557 - val_accuracy: 0.6650
    Epoch 43/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.6853 - accuracy: 0.7579 - val_loss: 1.0838 - val_accuracy: 0.6517
    Epoch 44/100
    29068/29068 [==============================] - 16s 545us/step - loss: 0.6777 - accuracy: 0.7651 - val_loss: 1.0893 - val_accuracy: 0.6601
    Epoch 45/100
    29068/29068 [==============================] - 16s 553us/step - loss: 0.6632 - accuracy: 0.7704 - val_loss: 1.0955 - val_accuracy: 0.6567
    Epoch 46/100
    29068/29068 [==============================] - 16s 552us/step - loss: 0.6484 - accuracy: 0.7715 - val_loss: 1.0742 - val_accuracy: 0.6563
    Epoch 47/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.6359 - accuracy: 0.7789 - val_loss: 1.0907 - val_accuracy: 0.6628
    Epoch 48/100
    29068/29068 [==============================] - 16s 556us/step - loss: 0.6296 - accuracy: 0.7821 - val_loss: 1.0694 - val_accuracy: 0.6663
    Epoch 49/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.6196 - accuracy: 0.7874 - val_loss: 1.0493 - val_accuracy: 0.6687
    Epoch 50/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.6082 - accuracy: 0.7932 - val_loss: 1.0703 - val_accuracy: 0.6622
    Epoch 51/100
    29068/29068 [==============================] - 16s 556us/step - loss: 0.5967 - accuracy: 0.7972 - val_loss: 1.0742 - val_accuracy: 0.6681
    Epoch 52/100
    29068/29068 [==============================] - 16s 558us/step - loss: 0.5761 - accuracy: 0.8020 - val_loss: 1.0736 - val_accuracy: 0.6641
    Epoch 53/100
    29068/29068 [==============================] - 16s 554us/step - loss: 0.5720 - accuracy: 0.8052 - val_loss: 1.1202 - val_accuracy: 0.6666
    Epoch 54/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.5669 - accuracy: 0.8048 - val_loss: 1.1161 - val_accuracy: 0.6607
    Epoch 55/100
    29068/29068 [==============================] - 16s 552us/step - loss: 0.5470 - accuracy: 0.8134 - val_loss: 1.1195 - val_accuracy: 0.6749
    Epoch 56/100
    29068/29068 [==============================] - 16s 552us/step - loss: 0.5418 - accuracy: 0.8176 - val_loss: 1.1947 - val_accuracy: 0.6749
    Epoch 57/100
    29068/29068 [==============================] - 16s 559us/step - loss: 0.5310 - accuracy: 0.8204 - val_loss: 1.1357 - val_accuracy: 0.6663
    Epoch 58/100
    29068/29068 [==============================] - 16s 550us/step - loss: 0.5295 - accuracy: 0.8205 - val_loss: 1.1116 - val_accuracy: 0.6721
    Epoch 59/100
    29068/29068 [==============================] - 16s 556us/step - loss: 0.5183 - accuracy: 0.8235 - val_loss: 1.1347 - val_accuracy: 0.6697
    Epoch 60/100
    29068/29068 [==============================] - 16s 550us/step - loss: 0.5101 - accuracy: 0.8288 - val_loss: 1.1133 - val_accuracy: 0.6706
    Epoch 61/100
    29068/29068 [==============================] - 15s 529us/step - loss: 0.4927 - accuracy: 0.8350 - val_loss: 1.1275 - val_accuracy: 0.6663
    Epoch 62/100
    29068/29068 [==============================] - 15s 528us/step - loss: 0.4869 - accuracy: 0.8382 - val_loss: 1.1522 - val_accuracy: 0.6728
    Epoch 63/100
    29068/29068 [==============================] - 16s 543us/step - loss: 0.4814 - accuracy: 0.8370 - val_loss: 1.1664 - val_accuracy: 0.6734
    Epoch 64/100
    29068/29068 [==============================] - 16s 551us/step - loss: 0.4712 - accuracy: 0.8415 - val_loss: 1.1616 - val_accuracy: 0.6687
    Epoch 65/100
    29068/29068 [==============================] - 16s 550us/step - loss: 0.4640 - accuracy: 0.8476 - val_loss: 1.1591 - val_accuracy: 0.6721
    Epoch 66/100
    29068/29068 [==============================] - 16s 552us/step - loss: 0.4482 - accuracy: 0.8500 - val_loss: 1.2533 - val_accuracy: 0.6715
    Epoch 67/100
    29068/29068 [==============================] - 16s 557us/step - loss: 0.4549 - accuracy: 0.8511 - val_loss: 1.1381 - val_accuracy: 0.6845
    Epoch 68/100
    29068/29068 [==============================] - 16s 554us/step - loss: 0.4472 - accuracy: 0.8526 - val_loss: 1.2276 - val_accuracy: 0.6681
    Epoch 69/100
    29068/29068 [==============================] - 16s 554us/step - loss: 0.4406 - accuracy: 0.8550 - val_loss: 1.0978 - val_accuracy: 0.6749
    Epoch 70/100
    29068/29068 [==============================] - 16s 550us/step - loss: 0.4347 - accuracy: 0.8560 - val_loss: 1.2091 - val_accuracy: 0.6768
    Epoch 71/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.4322 - accuracy: 0.8598 - val_loss: 1.2131 - val_accuracy: 0.6734
    Epoch 72/100
    29068/29068 [==============================] - 16s 553us/step - loss: 0.4138 - accuracy: 0.8666 - val_loss: 1.2282 - val_accuracy: 0.6703
    Epoch 73/100
    29068/29068 [==============================] - 16s 553us/step - loss: 0.4018 - accuracy: 0.8673 - val_loss: 1.1906 - val_accuracy: 0.6768
    Epoch 74/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.4006 - accuracy: 0.8700 - val_loss: 1.2685 - val_accuracy: 0.6712
    Epoch 75/100
    29068/29068 [==============================] - 16s 554us/step - loss: 0.3859 - accuracy: 0.8742 - val_loss: 1.2888 - val_accuracy: 0.6669
    Epoch 76/100
    29068/29068 [==============================] - 16s 547us/step - loss: 0.3906 - accuracy: 0.8703 - val_loss: 1.1981 - val_accuracy: 0.6737
    Epoch 77/100
    29068/29068 [==============================] - 16s 543us/step - loss: 0.3817 - accuracy: 0.8755 - val_loss: 1.3098 - val_accuracy: 0.6768
    Epoch 78/100
    29068/29068 [==============================] - 16s 558us/step - loss: 0.3722 - accuracy: 0.8769 - val_loss: 1.2980 - val_accuracy: 0.6783
    Epoch 79/100
    29068/29068 [==============================] - 16s 552us/step - loss: 0.3706 - accuracy: 0.8803 - val_loss: 1.3435 - val_accuracy: 0.6721
    Epoch 80/100
    29068/29068 [==============================] - 16s 551us/step - loss: 0.3671 - accuracy: 0.8801 - val_loss: 1.3179 - val_accuracy: 0.6709
    Epoch 81/100
    29068/29068 [==============================] - 16s 556us/step - loss: 0.3648 - accuracy: 0.8819 - val_loss: 1.2609 - val_accuracy: 0.6718
    Epoch 82/100
    29068/29068 [==============================] - 16s 552us/step - loss: 0.3691 - accuracy: 0.8812 - val_loss: 1.2705 - val_accuracy: 0.6752
    Epoch 83/100
    29068/29068 [==============================] - 16s 556us/step - loss: 0.3666 - accuracy: 0.8840 - val_loss: 1.2251 - val_accuracy: 0.6721
    Epoch 84/100
    29068/29068 [==============================] - 16s 554us/step - loss: 0.3462 - accuracy: 0.8880 - val_loss: 1.3033 - val_accuracy: 0.6737
    Epoch 85/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.3432 - accuracy: 0.8916 - val_loss: 1.3546 - val_accuracy: 0.6830
    Epoch 86/100
    29068/29068 [==============================] - 16s 550us/step - loss: 0.3416 - accuracy: 0.8938 - val_loss: 1.2883 - val_accuracy: 0.6659
    Epoch 87/100
    29068/29068 [==============================] - 16s 552us/step - loss: 0.3295 - accuracy: 0.8960 - val_loss: 1.3755 - val_accuracy: 0.6598
    Epoch 88/100
    29068/29068 [==============================] - 16s 558us/step - loss: 0.3287 - accuracy: 0.8957 - val_loss: 1.3392 - val_accuracy: 0.6681
    Epoch 89/100
    29068/29068 [==============================] - 16s 553us/step - loss: 0.3222 - accuracy: 0.8967 - val_loss: 1.3773 - val_accuracy: 0.6718
    Epoch 90/100
    29068/29068 [==============================] - 16s 556us/step - loss: 0.3262 - accuracy: 0.8955 - val_loss: 1.2527 - val_accuracy: 0.6765
    Epoch 91/100
    29068/29068 [==============================] - 16s 557us/step - loss: 0.3216 - accuracy: 0.8967 - val_loss: 1.3114 - val_accuracy: 0.6659
    Epoch 92/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.3122 - accuracy: 0.9013 - val_loss: 1.4067 - val_accuracy: 0.6752
    Epoch 93/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.3121 - accuracy: 0.9020 - val_loss: 1.3723 - val_accuracy: 0.6697
    Epoch 94/100
    29068/29068 [==============================] - 16s 553us/step - loss: 0.3052 - accuracy: 0.9035 - val_loss: 1.3329 - val_accuracy: 0.6712
    Epoch 95/100
    29068/29068 [==============================] - 16s 556us/step - loss: 0.3130 - accuracy: 0.9019 - val_loss: 1.3798 - val_accuracy: 0.6737
    Epoch 96/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.2925 - accuracy: 0.9085 - val_loss: 1.3849 - val_accuracy: 0.6755
    Epoch 97/100
    29068/29068 [==============================] - 16s 544us/step - loss: 0.3084 - accuracy: 0.9044 - val_loss: 1.5044 - val_accuracy: 0.6762
    Epoch 98/100
    29068/29068 [==============================] - 16s 559us/step - loss: 0.3033 - accuracy: 0.9052 - val_loss: 1.3532 - val_accuracy: 0.6681
    Epoch 99/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.2967 - accuracy: 0.9068 - val_loss: 1.4557 - val_accuracy: 0.6721
    Epoch 100/100
    29068/29068 [==============================] - 16s 555us/step - loss: 0.2999 - accuracy: 0.9058 - val_loss: 1.4230 - val_accuracy: 0.6697





    <keras.callbacks.callbacks.History at 0x7f589c68b690>




```python
#saving the  model to be used later
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
print("Saved model to disk")
```

    Saved model to disk


### Trainiertes Modell laden

In diesem Abschnitt wird das traniierte Modell geladen und die zu Beginn definierten Testdaten werden zu Performancemessung verwendet.


```python
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")
```

    Loaded model from disk



```python
truey=[]
predy=[]
x = np.load('./modXtest.npy')
y = np.load('./modytest.npy')

yhat= loaded_model.predict(x)
yh = yhat.tolist()
yt = y.tolist()
count = 0

for i in range(len(y)):
    yy = max(yh[i])
    yyt = max(yt[i])
    predy.append(yh[i].index(yy))
    truey.append(yt[i].index(yyt))
    if(yh[i].index(yy)== yt[i].index(yyt)):
        count+=1

acc = (count/len(y))*100
print(acc)
```

    65.08776818055169


### Convolution Matrix plotten


```python
cm = confusion_matrix(truey, predy)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
title='Confusion matrix'

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
```


![png](output_19_0.png)


