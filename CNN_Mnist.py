# -*- coding: utf-8 -*-


from __future__ import print_function
 
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.datasets import mnist

def CNN_Onehot(x_train,x_test,batch_size,num_classes,epochs,y_train,y_test):
    model = Sequential()
    model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
   
    model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10,activation='softmax'))
    
    
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
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
    plt.tight_layout()
    plt.show()
batch_size = 256 # 批次大小
num_classes = 10 # 類別大小
epochs = 100 # 訓練迭代次數
(x_train, y_train), (x_test, y_test) = mnist.load_data()# 分割訓練集資料與測試集資料

#調整目標樣本型態，訓練集資料
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
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
    CNN_Onehot(x_train,x_test,batch_size,num_classes,epochs,y_train,y_test)