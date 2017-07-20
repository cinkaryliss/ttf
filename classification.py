"""
load_self_data
    自作のデータを学習データとして取り込む。
    ファイルはクラスごとにフォルダ分けされている必要があり、
    カレントディレクトリのdata直下に格納されている必要がある。
    フォルダの名前はtrain_img_dirsに格納されている名前と一緒にすること。

load_mnist
    MNISTデータを学習データとして取り込む。
    テスト用として使用。

mk_model
    モデルを作成する関数。
    ここでの記述方法はKerasとほぼ同じ(一部異なる点がある)。
    引数として構造の名前を与えてやることでそのモデルを返す。

visualize_filters
    重みを可視化する関数。
    現在のところ、畳み込み層の第1層と第2層の1つ目のチャネルのみを表示する。
"""

import numpy as np
import datetime, time, os, cv2, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.contrib.keras.python.keras.datasets import mnist
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras.utils import np_utils, vis_utils
from tensorflow.contrib.keras.python.keras import backend as K

def load_self_data(classes=5, img_size=28):
    train_img_dirs = ['hard-kenzen', 'hard-koshi', 'soft-kenzen', 'kaoku', 'ground']
    train_image = []
    train_label = []
    test_image = []
    test_label = []

    for (i,d) in enumerate(train_img_dirs):
        #path以下のファイル名を取得
        path = os.getcwd() + '/data/' + d
        files = []
        for x in os.listdir(path):
            if not os.path.isdir(path + x) and x != 'number.txt' and x != '.DS_Store': #ディレクトリと指定したファイルを除く
                files.append(x)

        total = len(files)
        thresh = round(total*0.8) #20%のデータをテスト用として取り置き
        print('クラス{0} 総数:{1}\t訓練データ:{2}\tテストデータ:{3}'.format(i+1, total, thresh, total-thresh))

        k = 0
        flag = False

        for f in files:
            if not flag: #トレーニングデータ
                # 画像読み込み
                img = cv2.imread(os.getcwd() + '/data/' + d + '/' + f)
                img = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_LINEAR) #バイリニア法
                img = img.astype('float32')/255.0 #normalization
                train_image.append(img)

                # one_hot_vectorを作りラベルとして追加
                tmp = np.zeros(classes)
                tmp[i] = 1
                train_label.append(tmp)

                k += 1;
                if k == thresh:
                    flag = True

            else: #テストデータ
                # 画像読み込み
                img = cv2.imread(os.getcwd() + '/data/' + d + '/' + f)
                img = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_LINEAR) #バイリニア法
                img = img.astype('float32')/255.0 #normalization
                test_image.append(img)

                # one_hot_vectorを作りラベルとして追加
                tmp = np.zeros(classes)
                tmp[i] = 1
                test_label.append(tmp)

    #numpy配列に変換
    train_image = np.asarray(train_image) #(total*0.8,28,28,3)
    train_label = np.asarray(train_label) #(total*0.8,5)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)

    #かさ増しをするならここ
    #コントラスト正規化などもここ

    return train_image, train_label, test_image, test_label, 'aerial', classes, img_size

def load_mnist(classes=10): #MNIST(テスト用)
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

    return x_train, y_train, x_test, y_test, 'mnist', classes, 28

def mk_model(arch, data_type, classes):
    if data_type == 'mnist':
        channels = 1
    elif data_type == 'aerial':
        channels = 3

    if arch == 'lenet-5':
        model = Sequential() #モデルの初期化

        #畳み込み第１層
        model.add(Conv2D(32, 5, padding='same', input_shape=(28,28,channels))) #output_shape=(None,28,28,32)
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
        model.add(Dense(classes)) #output_shape=(None,classes)
        model.add(Activation('softmax'))

        return model

    elif arch == 'alexnet':
        model = Sequential() #モデルの初期化

        #alexnetを記述

def visualize_filters(model, title):
    W1 = model.layers[0].get_weights()[0] #(5,5,1,32)
    W1 = W1.transpose(3,2,0,1) #(32,1,5,5)

    W2 = model.layers[3].get_weights()[0] #(5,5,32,64)
    W2 = W2.transpose(3,2,0,1) #(64,32,5,5)

    scaler = MinMaxScaler(feature_range=(0,255)) #正規化用のフィルタ x_i_new = ((x_i-x_min)/(x_max-x_min))*255

    plt.figure()
    plt.suptitle('W1-channel1 '+title)
    for i in range(W1.shape[0]):
        im = W1[i,0]
        im = scaler.fit_transform(im) #normalization

        plt.subplot(4,8,i+1)
        plt.axis('off')
        plt.imshow(im, cmap='gray')
    plt.show()

    plt.figure()
    plt.suptitle('W2-channel1 '+title)
    for i in range(W2.shape[0]):
        im = W2[i,0]
        im = scaler.fit_transform(im) #normalization

        plt.subplot(8,8,i+1)
        plt.axis('off')
        plt.imshow(im, cmap='gray')
    plt.show()

if __name__=='__main__':
    todaydetail = datetime.datetime.today() #現在日時を取得
    filename = todaydetail.strftime('%Y_%m_%d-%H_%M_%S')
    start = time.time()

    #np.random.seed(1337) #for reproducibility
    batch_size = 50
    nb_epoch = 1

    x_train, y_train, x_test, y_test, data_type, classes, img_size = load_self_data()
    arch = 'lenet-5'
    model = mk_model(arch, data_type, classes)
    model.summary() #check model configuration

    #sys.exit()

    #visualize_filters(model, 'before') #重みの可視化

    #作成したモデルのアーキテクチャをnetwork.pngに出力
    vis_utils.plot_model(model, to_file='network-'+filename+'.png', show_shapes=True, show_layer_names=True)

    #学習プロセスの設定
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    #モデルの学習
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test))

    #visualize_filters(model, 'after') #重みの可視化

    #モデルの評価
    print('Evaluate')
    score = model.evaluate(x_test, y_test, verbose=1)

    print('\n\nTest score : {:>.4f}'.format(score[0])) #loss
    print('Test accuracy : {:>.4f}'.format(score[1]))

    elapsed_time = time.time() - start
    print('Time : {:>.4f} [sec]'.format(elapsed_time))

    #各種設定値等をテキストファイルに出力
    f = open('result-'+filename+'.txt', 'w')
    f.write('classes\t\t:\t{}'.format(classes))
    f.write('\nimg size\t:\t{0}x{1}'.format(img_size, img_size))
    f.write('\nbatch size\t:\t{}'.format(batch_size))
    f.write('\nepochs\t\t:\t{}'.format(nb_epoch))
    f.write('\narchitecture\t:\t{}'.format(arch))
    f.write('\noptimizer\t:\t{}'.format('Adam'))
    f.write('\nTest loss\t:\t{:>.4f}'.format(score[0]))
    f.write('\nTest accuracy\t:\t{:>.4f}'.format(score[1]))
    f.write('\nProcess time\t:\t{:>.4f}'.format(elapsed_time))
    f.close()

    K.clear_session() #バックエンド(TensorFlow)が使用していたリソースを解放
