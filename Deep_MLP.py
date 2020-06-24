# -*- coding: utf-8 -*-
from __future__ import print_function #相容python 2.X的print函數
#導入函式庫
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

def deep_MLP(x_train,x_test,batch_size,num_classes,epochs,y_train,y_test):
    model = Sequential()

    #Hidden_layer1+dropout layer
    model.add(Dense(512, activation='relu', input_shape=(784,),name='Hidden_layer1'))
    model.add(Dropout(0.5))
    
    #Hidden_layer2+dropout layer
    model.add(Dense(512, activation='relu',name='Hidden_layer2'))
    model.add(Dropout(0.5))
    
    #Hidden_layer3+dropout layer
    model.add(Dense(512, activation='relu',name='Hidden_layer3'))
    model.add(Dropout(0.5))
    
    #Hidden_layer4+dropout layer
    model.add(Dense(512, activation='relu',name='Hidden_layer4'))
    model.add(Dropout(0.5))
    
    #Hidden_layer5+dropout layer
    model.add(Dense(512, activation='relu',name='Hidden_layer5'))
    model.add(Dropout(0.5))
    
    
    #輸入至softmax分類器進行分類
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    #由於目標值為多種分類形式，loss 函數採用categorical_crossentropy，在優化器部分使用adam
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    #調用模型進行訓練
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
    
    loss,acc = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', loss)
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
    
    
batch_size = 256 # 批次大小
num_classes = 10 # 類別大小
epochs = 100 # 訓練迭代次數

(x_train, y_train), (x_test, y_test) = mnist.load_data()# 分割訓練集資料與測試集資料

#調整目標樣本型態，訓練集資料
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 轉換類別向量為二進制分類
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if __name__ == "__main__":
    deep_MLP(x_train,x_test,batch_size,num_classes,epochs,y_train,y_test)