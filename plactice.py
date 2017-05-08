import numpy as np
import time

from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras.utils import np_utils
from tensorflow.contrib.keras.python.keras import backend as K

def load_data(nb_classes=10):
    #the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32')
    x_test = x_test.reshape(10000, 784).astype('float32')
    #normalization(0〜255の値から0〜1に変換)
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #convert class vectors to binary class matrices(1 of nb_classesのベクトルに変換)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    return x_train, y_train, x_test, y_test

def mk_model():
    model = Sequential() #モデルの初期化
    model.add(Dense(512, input_dim=784)) #入力ー７８４次元、出力ー５１２次元
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

if __name__=='__main__':
    start = time.time()
    np.random.seed(1337) #for reproducibility
    batch_size = 128
    nb_epoch = 2

    x_train, y_train, x_test, y_test = load_data()
    model = mk_model()
    model.summary() #check model configuration

    #学習プロセスの設定
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    #モデルの学習
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test))

    #モデルの評価
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\nTest score : {:>.4f}'.format(score[0])) #loss
    print('Test accuracy : {:>.4f}'.format(score[1]))

    elapsed_time = time.time() - start
    print('Time : {:>.4f}'.format(elapsed_time) + '[sec]')

    K.clear_session()
