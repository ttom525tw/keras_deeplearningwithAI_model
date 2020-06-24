# -*- coding: utf-8 -*-
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import GRU
from keras.datasets import imdb
import matplotlib.pyplot as plt


def Imdb_GRU(x_train, y_train,x_test,y_test,batch_size,max_features):
    #模型建置
    model = Sequential()
    model.add(Embedding(max_features, 128))
    #word-embedding層
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
    #GRU層
    model.add(Dense(1, activation='sigmoid'))
    # 輸出層，透過sigmoid分類器
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #開始訓練
    print('Train...')
    history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=15,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    #透過matplot繪圖顯示訓練過程
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
            
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()
    
max_features = 20000 #特徵大小
maxlen = 80 #序列長度
batch_size = 1024 #批次大小

#調整目標樣本型態，訓練集資料
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('Build model...')   
 
if __name__ == "__main__":
   Imdb_GRU(x_train, y_train,x_test,y_test,batch_size,max_features)