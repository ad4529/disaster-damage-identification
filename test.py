import os.path

from keras.callbacks import TensorBoard, LambdaCallback
from keras.layers import Conv2D, Softmax, MaxPool2D
from keras.layers import Flatten, Dense
from keras.models import Sequential, load_model, Model

from dataset import DataSet

MODEL_SAVE_PATH = './model/model.h5'

model: Model = None


def get_model(input_shape):
    if not os.path.isdir('./model'):
        os.mkdir('model')
    try:
        if os.path.isfile(MODEL_SAVE_PATH):
            m = load_model(MODEL_SAVE_PATH)
            print('loaded model')
            return m
    except:
        print('Could not load saved model. Creating new.')
    m = Sequential()
    m.add(Conv2D(64, kernel_size=(11, 11), activation='relu', input_shape=input_shape, padding='same'))
    m.add(MaxPool2D((2, 2), (2, 2)))
    m.add(Conv2D(128, kernel_size=(11, 11), activation='relu', input_shape=input_shape, padding='same'))
    m.add(MaxPool2D((2, 2), (2, 2)))
    m.add(Conv2D(256, kernel_size=(11, 11), activation='relu', input_shape=input_shape, padding='same'))
    m.add(MaxPool2D((2, 2), (2, 2)))
    m.add(Conv2D(512, kernel_size=(11, 11), activation='relu', input_shape=input_shape, padding='same'))
    m.add(MaxPool2D((2, 2), (2, 2)))
    m.add(Conv2D(512, kernel_size=(11, 11), activation='relu', input_shape=input_shape, padding='same'))
    m.add(MaxPool2D((2, 2), (2, 2)))
    # m.add(Conv2D(512, kernel_size=(11, 11), activation='relu', input_shape=input_shape, padding='same'))
    # m.add(MaxPool2D((2, 2), (2, 2)))
    m.add(Flatten())
    # m.add(Dense(4096, activation='relu'))
    m.add(Dense(6, activation='relu'))
    m.add(Softmax())
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


def save_model(epoch, logs):
    global model
    if epoch % 100 == 0:
        print('saving model...')
        model.save(MODEL_SAVE_PATH)


def main():
    global model
    d = DataSet()
    d.load_data(batch_size=32)
    input_shape = (224, 224, 3)
    model = get_model(input_shape)
    model.summary()
    # exit(0)
    tensorboard_callback = TensorBoard('./logs')
    checkpoint = LambdaCallback(on_epoch_begin=save_model)
    callback_list = [tensorboard_callback, checkpoint]
    model.fit_generator(d.train_generator, steps_per_epoch=16, epochs=1000, callbacks=callback_list,
                        use_multiprocessing=True)


if __name__ == '__main__':
    main()
