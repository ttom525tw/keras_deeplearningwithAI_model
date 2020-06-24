# -*- coding: utf-8 -*-
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
def Deep_autoencoder(x_train,x_test):
    
    input_img = Input(shape=(784,))
    
    encoded = Dense(units=128, activation='relu')(input_img)
    encoded = Dense(units=64, activation='relu')(encoded)
    encoded = Dense(units=32, activation='relu')(encoded)
    decoded = Dense(units=64, activation='relu')(encoded)
    decoded = Dense(units=128, activation='relu')(decoded)
    decoded = Dense(units=784, activation='sigmoid')(decoded)
    autoencoder = Model(input_img, decoded)
    
    
    
    
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    
    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    
    
    #encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)
    
        
    
    
    
    n = 10  #輸出10個數字
    plt.figure(figsize=(20, 4))
    for i in range(n):
        #輸出原本的圖片
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        #輸出模型結果圖片
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

(x_train, _), (x_test, _) = mnist.load_data()
    
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
if __name__ == "__main__":
    Deep_autoencoder(x_train,x_test)