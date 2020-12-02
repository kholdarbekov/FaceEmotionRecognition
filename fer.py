import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model

num_labels = 7
batch_size = 128
epochs = 250
width, height = 48, 48

data = pd.read_csv("data/fer2013.csv")
pixels = data['pixels'].tolist()

faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split()]
    face = np.asarray(face).reshape(width, height)

    face = face / 255.0
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)

emotions = pd.get_dummies(data['emotion']).values

X_train = faces[:28622]
X_test = faces[28622:]
y_train = emotions[:28622]
y_test = emotions[28622:]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

model = tf.keras.Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(width, height, 1), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(1024, activation= 'relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(Dropout(0.4))
model.add(Dense(512, activation= 'relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(Dropout(0.3))
model.add(Dense(128, activation= 'relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(random.random()*100))))
model.add(Dropout(0.2))
model.add(Dense(num_labels, activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta=0.0001, patience=10, verbose=1, min_lr=0.000001)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
    )


datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
          steps_per_epoch=len(X_train) / batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks = [lr_reducer])


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 3])
plt.legend(loc='upper right')
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
model.save('models/fer_model.h5')

#load and finetune the pre-trained model
model2 = load_model('models/fer_model.h5')

model2.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00003, decay=1e-6, momentum=0.9, nesterov=True),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=6, verbose=1, min_delta=0.00001, min_lr=0.0000001)

history2 = model2.fit(X_train, y_train, batch_size=batch_size, epochs=100, validation_data=(X_val, y_val), callbacks=[lr_reducer])

model2.save('models/fer_model_finetuned.h5')
model2.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)