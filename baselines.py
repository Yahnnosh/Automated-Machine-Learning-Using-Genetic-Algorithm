from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import time
#%%

# Loading data + preprocessing
print('Loading data')
t0 = time.time()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)   # one-hot encoding
y_test = to_categorical(y_test)  # one-hot encoding
print('Finished loading data ({}s)\n'.format(round(time.time() - t0, 3)))
#%%

# 1) Torben network (~0.98)
hard_baseline_model = models.Sequential()
hard_baseline_model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
hard_baseline_model.add(layers.BatchNormalization())
hard_baseline_model.add(layers.MaxPool2D(strides=(2, 2)))

hard_baseline_model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
hard_baseline_model.add(layers.BatchNormalization())
hard_baseline_model.add(layers.MaxPool2D(strides=(2, 2)))

hard_baseline_model.add(layers.Flatten())
hard_baseline_model.add(layers.Dropout(0.4))

hard_baseline_model.add(layers.Dense(128, activation='relu'))
hard_baseline_model.add(layers.Dropout(0.3))
hard_baseline_model.add(layers.Dense(10, activation='softmax'))

#baseline_model.summary()
hard_baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hard_baseline_model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))
#%%

# 2) Simple multilayer perceptron (~0.93)
low_baseline_model = models.Sequential()
low_baseline_model.add(layers.Flatten())
low_baseline_model.add(layers.Dense(32, activation='relu'))
low_baseline_model.add(layers.Dense(10, activation='softmax'))

low_baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
low_baseline_model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))

#%%

# 3) Simple CNN (~0.97)
model = models.Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))
