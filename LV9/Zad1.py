import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorboard import notebook


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

#X_train_n = X_train_n[0:25000,:,:,:] #izdvajanje polovine skupa podataka za ucenje u svrhu zadatka


# 1-od-K kodiranje
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


# CNN mreza, broj parametara = 1 122 758
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# Učitavanje log datoteke
logdir = "logs/"

# Pokretanje Tensorboarda u notebooku
notebook.start("--logdir " + logdir)

my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_earlyStop',  
                                update_freq = 100),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=40) 
]

optimizer = keras.optimizers.Adam() 
model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])


model.fit(X_train_n,
            y_train,
            epochs = 40,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}%')


