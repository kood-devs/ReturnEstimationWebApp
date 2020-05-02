"""
    Google (2015)
    URL: https://github.com/corrieelston/datalab/blob/master/FinancialTimeSeriesTensorFlow.ipynb
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
from keras import models, layers

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# CONST
TRAINING_START = datetime(2000, 1, 1)
TRAINING_END = datetime(2015, 12, 31)
TEST_START = datetime(2016, 1, 1)
TEST_END = datetime(2019, 10, 1)
EPOCH = 50
BATCH_SIZE = 128

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


# 学習結果をプロット
def show_learning_process(history):
    history_dict = history.history

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def get_log_return(start_date, end_date, index_list, same_index_list, number_of_shift=3):
    # 終値を取得
    closing_data = pd.DataFrame()
    for name in index_list:
        closing_data[name[0]] = pdr.DataReader(
            name[1], 'yahoo', start_date, end_date)['Close']
    closing_data = closing_data.fillna(method='ffill')  # 休場日は前日終値を横置き

    # 一応データを保存しておく
    closing_data.to_csv('result/index.csv')

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
    train_test_data.to_csv(
        'result/learning_data_{}.csv'.format(number_of_shift))
    x_val = np.array(train_test_data.iloc[:, 1:])
    y_val = np.array(train_test_data.iloc[:, 0]).reshape(-1)
    return x_val, y_val


def learn_test_models(x_train, y_train, x_test, y_test):
    # 様々なモデルで訓練を実施、テストデータのパフォーマンスを計算
    result = pd.DataFrame()  # 結果の出力用

    # ------------------------------------------
    #  DNN
    # ------------------------------------------
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu',
                           input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE,
                        validation_data=(x_test, y_test))
    show_learning_process(history)

    acc = sum([1 if val >= 0.5 else 0 for val in model.predict(x_train)] == y_train)
    acc /= y_train.shape[0]
    val_acc = sum(
        [1 if val >= 0.5 else 0 for val in model.predict(x_test)] == y_test)
    val_acc /= y_test.shape[0]
    result['DNN'] = np.array([acc, val_acc])

    # ------------------------------------------
    #  二項ロジット
    # ------------------------------------------
    model = LogisticRegression(penalty='none', max_iter=1e10, tol=1e-10)
    model.fit(x_train, y_train)

    acc = sum(model.predict(x_train) == y_train) / y_train.shape[0]
    val_acc = sum(model.predict(x_test) == y_test) / y_test.shape[0]
    result['BinomialLogit'] = np.array([acc, val_acc])

    # ------------------------------------------
    #  サポートベクトルマシン
    # ------------------------------------------
    # 線形SVC
    model = SVC(kernel='linear', random_state=1, C=1.0)
    model.fit(x_train, y_train)

    acc = sum(model.predict(x_train) == y_train) / y_train.shape[0]
    val_acc = sum(model.predict(x_test) == y_test) / y_test.shape[0]
    result['LinearSVC'] = np.array([acc, val_acc])

    # カーネルSVC
    model = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
    model.fit(x_train, y_train)

    acc = sum(model.predict(x_train) == y_train) / y_train.shape[0]
    val_acc = sum(model.predict(x_test) == y_test) / y_test.shape[0]
    result['KernelSVC'] = np.array([acc, val_acc])

    # ------------------------------------------
    #  決定木
    # ------------------------------------------
    model = DecisionTreeClassifier(
        criterion='gini', max_depth=4, random_state=1)
    model.fit(x_train, y_train)

    acc = sum(model.predict(x_train) == y_train) / y_train.shape[0]
    val_acc = sum(model.predict(x_test) == y_test) / y_test.shape[0]
    result['DecisionTree'] = np.array([acc, val_acc])

    # 結果の出力
    result = result.rename(index={0: 'acc', 1: 'val_acc'})
    print(result)
    result.to_csv('result/result_several.csv')


def learn_test_rnn(x_train, y_train, x_test, y_test):
    # RNNで訓練を実施、結果を出力
    result = pd.DataFrame()

    # モデルを構築
    model = models.Sequential()
    model.add(layers.LSTM(32, batch_input_shape=(
        None, x_train.shape[1], x_train.shape[2]), activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE,
                        validation_data=(x_test, y_test))
    show_learning_process(history)

    # 結果を算出
    acc = sum([1 if val >= 0.5 else 0 for val in model.predict(x_train)] == y_train)
    acc /= y_train.shape[0]
    val_acc = sum(
        [1 if val >= 0.5 else 0 for val in model.predict(x_test)] == y_test)
    val_acc /= y_test.shape[0]

    result['RNN'] = np.array([acc, val_acc])
    result.to_csv('result/result_rnn.csv')


def main():
    # データを準備
    x_train, y_train = get_log_return(
        TRAINING_START, TRAINING_END, EXCHANGES_DEFINE, SAME_EXCHANGES_DEFINE)
    x_test, y_test = get_log_return(
        TEST_START, TEST_END, EXCHANGES_DEFINE, SAME_EXCHANGES_DEFINE)

    # 学習を実施・結果を出力
    learn_test_models(x_train, y_train, x_test, y_test)

    # RNNによる学習
    x_train, y_train = get_log_return(
        TRAINING_START, TRAINING_END, EXCHANGES_DEFINE, SAME_EXCHANGES_DEFINE, number_of_shift=1)
    x_test, y_test = get_log_return(
        TEST_START, TEST_END, EXCHANGES_DEFINE, SAME_EXCHANGES_DEFINE, number_of_shift=1)

    # RNN学習用にreshape
    length_of_sequences = 10
    x_val = []
    for i in range(x_train.shape[0] - length_of_sequences + 1):
        x_val.append(x_train[i:i+length_of_sequences])
    x_train = np.array(x_val)

    x_val = []
    for i in range(x_test.shape[0] - length_of_sequences + 1):
        x_val.append(x_test[i:i+length_of_sequences])
    x_test = np.array(x_val)

    y_train = y_train[length_of_sequences-1:]
    y_test = y_test[length_of_sequences-1:]

    # 学習を実施・結果を出力
    learn_test_rnn(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
