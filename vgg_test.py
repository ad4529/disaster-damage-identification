import numpy as np
import keras
from keras.layers import Conv2D, Softmax
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D
from dataset import DataSet
from keras.optimizers import Adam
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

if __name__ == '__main__':
    d = DataSet()
    d.load_data()

base_model=VGG16(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(6,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers:
    layer.trainable=True


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit_generator(d.train_generator, steps_per_epoch=32, epochs=45, verbose=1)




