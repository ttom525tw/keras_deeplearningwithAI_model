# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras import  initializers,regularizers,activations,constraints
from keras.engine.topology import Layer,InputSpec
from keras.layers import LSTM
import matplotlib.pyplot as plt

class SelfAttention(Layer):
#注意力模型(Attention layer:Self Attention)
    def __init__(self,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SelfAttention, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        time_steps = input_shape[1]
        dimensions = input_shape[1]

        self.attention = keras.models.Sequential(name='attention')
        
        self.attention.add(keras.layers.Dense(dimensions,
                                              input_shape=(
                                                  time_steps, dimensions,),
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint))
        self.attention.add(keras.layers.Activation(self.activation))
        self.attention.add(keras.layers.Dense(1,
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint))
        
        self.attention.add(keras.layers.Flatten())
        self.attention.add(keras.layers.Activation('softmax'))
        
        self.attention.add(keras.layers.RepeatVector(dimensions))
        
        self.attention.add(keras.layers.Permute([2, 1]))

        
        self.trainable_weights = self.attention.trainable_weights
        self.non_trainable_weights = self.attention.non_trainable_weights

        
        self.built = True

    def call(self, inputs):
        
        attention = self.attention(inputs)
        
        return keras.layers.Multiply()([inputs, attention])

    def compute_output_shape(self, input_shape):
        
        return input_shape

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        return dict(config)

def Lstm_Attention(x_train,x_test,batch_size,num_classes,epochs,y_train,y_test):
    #模型建置
    x = Input(shape=(row, col, pixel))
    # 輸入層
    encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
    # TimeDistributed層
    encoded_columns = LSTM(col_hidden)(encoded_rows)
    #LSTM隱藏層
    attention=SelfAttention()(encoded_columns)
    #注意力層
    prediction = Dense(num_classes, activation='softmax')(attention)
    #輸出層,透過softmax進行分類
    model = Model(x, prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    
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
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
batch_size = 256 # 批次大小
num_classes = 10 # 類別大小
epochs = 50 # 訓練迭代次數

row_hidden = 128
col_hidden = 128

#調整目標樣本型態，訓練集資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
row, col, pixel = x_train.shape[1:]

if __name__ == "__main__":
    Lstm_Attention(x_train,x_test,batch_size,num_classes,epochs,y_train,y_test)