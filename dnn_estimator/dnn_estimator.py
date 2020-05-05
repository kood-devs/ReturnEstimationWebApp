from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
from keras import models, layers

# CONST
EXCHANGES_DEFINE = [
    ['SP500', '^GSPC'],
    ['NYSE', '^NYA'],
    ['DOW', '^DJI'],
    ['NASDAQ', '^IXIC'],
    # ['FTSE', '^FTSE'],  # UK...現在は取得できず
    ['GDAXI', '^GDAXI'],  # Germany
    ['N225', '^N225'],
    ['HSI', '^HSI'],
    ['AORD', '^AORD'],
]

SAME_EXCHANGES_DEFINE = [  # 米欧のインデックスは前日終値～を使用
    ['SP500', '^GSPC'],
    ['NYSE', '^NYA'],
    ['DOW', '^DJI'],
    ['NASDAQ', '^IXIC'],
    ['GDAXI', '^GDAXI'],  # Germanyも当日のデータは使用せず
]


# 学習過程の画像を保存
def save_learning_process_img(history, img_name):
    history_dict = history.history
    print(history_dict)
    # print(dir(history_dict))
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('media/{}.jpg'.format(img_name))
    # plt.legend()
    # plt.show()


def get_log_return(start_date, end_date, index_list, same_index_list, number_of_shift=3):
    # 終値を取得
    closing_data = pd.DataFrame()
    for name in index_list:
        closing_data[name[0]] = pdr.DataReader(
            name[1], 'yahoo', start_date, end_date)['Close']
    closing_data = closing_data.fillna(method='ffill')  # 休場日は前日終値を横置き

    # 対数リターンに変換
    log_return_data = pd.DataFrame()
    for name in index_list:
        log_return_data[name[0] + '_log'] = np.log(
            closing_data[name[0]] / closing_data[name[0]].shift(1))

    # データ形式を整形
    train_test_data = pd.DataFrame()
    train_test_data['SP500_log_pos'] = (log_return_data['SP500_log'] > 0) * 1
    log_name_list = ['{}_log'.format(pair[0]) for pair in same_index_list]

    for col_name in log_return_data:
        if col_name in log_name_list:
            for i in range(number_of_shift):
                train_test_data[col_name +
                                str(i + 1)] = log_return_data[col_name].shift(i + 1)
        else:
            for i in range(number_of_shift):
                train_test_data[col_name +
                                str(i)] = log_return_data[col_name].shift(i)

    # 型を整えてデータを返却
    train_test_data = train_test_data.dropna()
    x_val = np.array(train_test_data.iloc[:, 1:])
    y_val = np.array(train_test_data.iloc[:, 0]).reshape(-1)
    return x_val, y_val


def learn_dnn(train_start, train_end, test_start, test_end, epoch, batch_size, img_name):
    # データを準備
    x_train, y_train = get_log_return(
        train_start, train_end, EXCHANGES_DEFINE, SAME_EXCHANGES_DEFINE)
    x_test, y_test = get_log_return(
        test_start, test_end, EXCHANGES_DEFINE, SAME_EXCHANGES_DEFINE)

    # 学習を実施
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu',
                           input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,
                        validation_data=(x_test, y_test))

    # 学習過程の保存
    save_learning_process_img(history, img_name)

    # 結果を出力
    acc = sum([1 if val >= 0.5 else 0 for val in model.predict(x_train)] == y_train)
    acc /= y_train.shape[0]
    val_acc = sum(
        [1 if val >= 0.5 else 0 for val in model.predict(x_test)] == y_test)
    val_acc /= y_test.shape[0]
    return acc, val_acc
