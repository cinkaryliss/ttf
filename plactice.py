import numpy as np
import time, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras.utils import np_utils, vis_utils
from tensorflow.contrib.keras.python.keras import backend as K

def load_data(nb_classes=10):
    #the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #この時点でx_train.shape=(60000,28,28), x_test.shape=(10000,28,28), y_train.shape=(60000,), y_test.shape=(10000,)
    x_train = x_train.reshape(-1,28,28,1).astype('float32') #-1はそれ以外に合わせるように合わせるという意味
    x_test = x_test.reshape(-1,28,28,1).astype('float32')
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

    #畳み込み第１層
    model.add(Conv2D(32, 5, padding='same', input_shape=(28,28,1))) #output_shape=(None(60000),28,28,32)
    #filters=32, kernel_size=(5,5), strides(1,1), use_bias=True
    #dilidation_rate(膨張率)=(1,1), kernel_initializer='glorot_uniform', bias_initializer='zeros'
    #padding='sane'は出力のshapeが入力と同じになるように調整
    #output_shape=(None(60000),28,28,32)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(padding='same')) #output_shape=(None,14,14,32)
    #pool_size=(2,2), strides(2,2)

    #畳み込み第２層
    model.add(Conv2D(64, 5, padding='same')) #output_shape=(None,14,14,64)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(padding='same')) #output_shape=(None,7,7,64)

    #平坦化
    model.add(Flatten()) #output_shape=(None,3136(7*7*64))

    #全結合第１層
    model.add(Dense(1024)) #output_shape=(None,1024)
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) #無視する割合を記述(例えば、0.2と記述した場合、80%の結合が残る)

    #全結合第２層
    model.add(Dense(10)) #output_shape=(None,10)
    model.add(Activation('softmax'))

    return model

def visualize_filters(model, title):
    W1 = model.layers[0].get_weights()[0] #(5,5,1,32)
    W1 = W1.transpose(3,2,0,1) #(32,1,5,5)

    W2 = model.layers[3].get_weights()[0] #(5,5,32,64)
    W2 = W2.transpose(3,2,0,1) #(64,32,5,5)

    scaler = MinMaxScaler(feature_range=(0,255)) #正規化用のフィルタ x_i_new = ((x_i-x_min)/(x_max-x_min))*255

    plt.figure()
    plt.suptitle('W1 '+title)
    for i in range(W1.shape[0]):
        im = W1[i,0]
        im = scaler.fit_transform(im) #normalization

        plt.subplot(4,8,i+1)
        plt.axis('off')
        plt.imshow(im, cmap='gray')
    plt.show()

    plt.figure()
    plt.suptitle('W2-1 '+title)
    for i in range(W2.shape[0]):
        im = W2[i,0]
        im = scaler.fit_transform(im) #normalization

        plt.subplot(8,8,i+1)
        plt.axis('off')
        plt.imshow(im, cmap='gray')
    plt.show()

if __name__=='__main__':
    start = time.time()
    np.random.seed(1337) #for reproducibility
    batch_size = 100
    nb_epoch = 2

    x_train, y_train, x_test, y_test = load_data()
    model = mk_model()
    model.summary() #check model configuration

    visualize_filters(model, 'before')

    vis_utils.plot_model(model, to_file='network.png', show_shapes=True, show_layer_names=True)
    #plot(model, to_file='model.png', show_shapes=True, show_layers_name=True)

    #学習プロセスの設定
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    #モデルの学習
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test))

    visualize_filters(model, 'after')

    #モデルの評価
    print('Evaluate')
    score = model.evaluate(x_test, y_test, verbose=1)
    print('\n\nTest score : {:>.4f}'.format(score[0])) #loss
    print('Test accuracy : {:>.4f}'.format(score[1]))

    elapsed_time = time.time() - start
    print('Time : {:>.4f} [sec]'.format(elapsed_time))

    K.clear_session() #バックエンド(TensorFlow)が使用していたリソースを解放
