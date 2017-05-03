import numpy as np

from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras.utils import np_utils
from tensorflow.contrib.keras.python.keras import backend as K

def load_data(nb_classes=10):
    #the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    return x_train, y_train, x_test, y_test

def mk_model():
    model = Sequential()
    model.add(Dense(512, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

if __name__=='__main__':
    np.random.seed(1337) #for reproducibility
    batch_size = 128
    nb_epoch = 20

    x_train, y_train, x_test, y_test = load_data()
    model = mk_model()
    model.summary() #check model configuration

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('\nTest score : {:>.4f}'.format(score[0]))
    print('Test accuracy : {:>.4f}'.format(score[1]))

    K.clear_session()
    #this statement is fixed the condition of ...
    #Exception ignored in: <bound methid BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7fb79a3fa550>>
    #...
    #AttributeError: 'NoneType' object has no attribute 'TF_NewStatus'
    #TensorFlow issue: Exception ignored in BaseSession.__del__ #3388
