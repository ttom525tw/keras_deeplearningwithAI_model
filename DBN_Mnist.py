# -*- coding: utf-8 -*-
from sklearn.neural_network import BernoulliRBM
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt

class DBN():

  def __init__(
    self,
    train_data,
    targets, 
    layers,
    outputs,
    rbm_lr,
    rbm_iters,
    rbm_dir=None,
    test_data = None,
    test_targets = None,    
    epochs = 50,
    fine_tune_batch_size = 128
     ):

    self.hidden_sizes = layers
    self.outputs = outputs
    self.targets = targets
    self.data = train_data

    if test_data is None:
      self.validate = False
    else:
      self.validate = True

    self.valid_data = test_data
    self.valid_labels = test_targets

    self.rbm_learning_rate = rbm_lr
    self.rbm_iters = rbm_iters

    self.epochs = epochs
    self.nn_batch_size = fine_tune_batch_size

    self.rbm_weights = []
    self.rbm_biases = []
    self.rbm_h_act = []

    self.model = None
    self.history = None

    

  def pretrain(self,save=True):
    
    visual_layer = self.data

    for i in range(len(self.hidden_sizes)):
      print("[DBN] Layer {} Pre-Training".format(i+1))

      rbm = BernoulliRBM(n_components = self.hidden_sizes[i], n_iter = self.rbm_iters[i], learning_rate = self.rbm_learning_rate[i],  verbose = True, batch_size = 128)
      rbm.fit(visual_layer)
      self.rbm_weights.append(rbm.components_)
      self.rbm_biases.append(rbm.intercept_hidden_)
      self.rbm_h_act.append(rbm.transform(visual_layer))

      visual_layer = self.rbm_h_act[-1]

    




  def finetune(self):
    model = Sequential()
    for i in range(len(self.hidden_sizes)):

      if i==0:
        model.add(Dense(self.hidden_sizes[i], activation='relu', input_dim=self.data.shape[1], name='rbm_{}'.format(i)))
      else:
        model.add(Dense(self.hidden_sizes[i], activation='relu', name='rbm_{}'.format(i)))


    model.add(Dense(self.outputs, activation='softmax'))
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    for i in range(len(self.hidden_sizes)):
      layer = model.get_layer('rbm_{}'.format(i))
      layer.set_weights([self.rbm_weights[i].transpose(),self.rbm_biases[i]])

    

    if self.validate:
      self.history = model.fit(x_train, y_train, 
                              epochs = self.epochs, 
                              batch_size = self.nn_batch_size,
                              validation_data=(self.valid_data, self.valid_labels),
                              )
      loss,acc = model.evaluate(x_test, y_test, verbose=0)
    
      print('Test loss:', loss)
      print('Test accuracy:', acc)
        
        #透過matplot繪圖顯示訓練過程
      plt.subplot(211)
      plt.plot(self.history.history['acc'])
      plt.plot(self.history.history['val_acc'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='best')
      plt.subplot(212)
      plt.plot(self.history.history['loss'])
      plt.plot(self.history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='best')
      plt.show()
    else:
       self.history = model.fit(x_train, y_train, 
                              epochs = self.epochs, 
                              batch_size = self.nn_batch_size,
                              )  
       loss,acc = model.evaluate(x_test, y_test, verbose=0)
    
       print('Test loss:', loss)
       print('Test accuracy:', acc)
        
        #透過matplot繪圖顯示訓練過程
       plt.subplot(211)
       plt.plot(self.history.history['acc'])
       plt.plot(self.history.history['val_acc'])
       plt.title('model accuracy')
       plt.ylabel('accuracy')
       plt.xlabel('epoch')
       plt.legend(['train', 'test'], loc='best')
       plt.subplot(212)
       plt.plot(self.history.history['loss'])
       plt.plot(self.history.history['val_loss'])
       plt.title('model loss')
       plt.ylabel('loss')
       plt.xlabel('epoch')
       plt.legend(['train', 'test'], loc='best')
       plt.show()
    self.model = model

  


  


      
   
num_classes=10     
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

if __name__ == '__main__':
  dbn = DBN(train_data = x_train, targets = y_train,
            test_data = x_test, test_targets = y_test,
            layers = [200],
            outputs = 10,
            rbm_iters = [40],
            rbm_lr = [0.01])
  dbn.pretrain(save=True)
  dbn.finetune()

 