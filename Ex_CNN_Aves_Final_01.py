#-*- coding: utf-8 -*-
#Alimentando seu próprio conjunto de dados no modelo da CNN em Keras

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils

import numpy as np

# Usa Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import theano
from PIL import Image
from numpy import *

import time


from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

from keras.layers import Input
from keras import backend as k

import collections

from sklearn.metrics import confusion_matrix


# Parâmetros de configuração
# -------------------------------------------
batch_size = 32
num_classes = 3
nb_epoch = 30
img_rows, img_cols = 120, 180
img_channels = 3
nb_filters = 512 # Corresponde ao número de mapas de características diferentes
nb_pool = 2
nb_conv = 3      # Stride é o número de pixels pelo qual nós deslizamos nossa matriz de filtros
# -------------------------------------------

# Dimensoes da imagem de entrada
# --------------------------------------------
input_img = Input(shape=(img_rows, img_cols, 3))
print("Forma da Imagem de entrada", k.int_shape(input_img))
# --------------------------------------------

# Ler os arquivos contendo as imagens
# -------------------------------------------
path1 = 'train'
#path1 = 'train2'
#path2 = 'img_cinza'
# -------------------------------------------

# Mostrar as imagens de cada Arquivo
# -------------------------------------------
listing = os.listdir(path1)
listing.sort()
num_amostras_s = size(listing)
print('\nUsando {} amostras da especie Dendrocygna Viduata para treinamento'.format(num_amostras_s))
# -------------------------------------------

# Converter as imagens para a escala de Cinza e armazenar elas em path2
# -------------------------------------------
#'''
#for i in listing:
#    im = Image.open(path1 + '/' + i)
#    img = im.resize((img_cols, img_rows))
#    gray = img.convert('L')
#    gray.save(path2 + '/' + i, 'jpeg')

#listing = os.listdir(path2)
#'''

# Teste dos parâmetros
# -------------------------------------------
im1 = array(Image.open('train' + '/' + listing[0]))  # abre uma imagem
m, n = im1.shape[0:2]   # obtem o tamanho da imagem
imnbr = len(listing)    # obtem o número de imagens
print('Numero de Linhas  ', m)
print('Numero de Colunas ', n)
print('Numero de imagens ', imnbr)
# -------------------------------------------

# Criar uma matriz para armazenar todas as imagens achatadas
# -------------------------------------------
#immatrix = numpy.array(Image.open(path + "/" + file).flatten()
immatrix = array([array(Image.open('train' + '/' + im2)).flatten()
	for im2 in listing], 'f')

# -------------------------------------------
label = np.ones((num_amostras_s,), dtype=int)

#'''
#label[0:349] = 0    # classe Garça Grande Branca - 349
#label[349:698] = 1  # classe Irire - 349
#label[698:] = 2     # classe Socozinho - 349

#'''
label[0:2000] = 0    # classe Garça Grande Branca - 2000
label[2000:4000] = 1  # classe Irire - 2000
label[4000:] = 2     # classe Socozinho - 2000




# O uso da funcao Shuffle torna o sistema mais estável
# Shuffle the dataset
X,Y = shuffle(immatrix, label, random_state=2)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify = Y)

classe_Y_train = collections.Counter(Y_train)
classe_Y_test = collections.Counter(Y_test)

print('\nUsando {} de amostras para o treinamento'.format(classe_Y_train))
print('Usando {} de amostras para o teste'.format(classe_Y_test))

#'''
# A função collections.Counter(Y_train) retorna a quantidade de elementos da mesma classe
collections.Counter(Y_train)
collections.Counter(Y_test)


#len(X)
#len(Y)
#len(X[0])
#len(X[2])
#print(np.matrix(X))
#print(np.matrix(Y))
#print('Y_test')
#print(np.matrix(Y_test))
#'''

print('\nForma de treinamento da amostra X:', X_train.shape)
print('Usando {} amostras para treinamento'.format(X_train.shape[0]))
print('Usando {} amostras para teste'.format(X_test.shape[0]))


if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('\nForma da amostra de treinamento:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

# Converte as classes dos vetores para matrizes de classes binárias
#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)


#'''
#i = 100
#plt.imshow(X_train[i, 0], interpolation='nearest')
#print("label : ", Y_train[i,:])
#'''

# -------------------------------------------
model = Sequential()

inicio = time.time()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation ='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (nb_conv, nb_conv), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (nb_conv, nb_conv), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (nb_conv, nb_conv), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (nb_conv, nb_conv), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (nb_conv, nb_conv), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.1))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#'''
#model.compile(loss='categorical_crossentropy',
#              optimizer=keras.optimizers.rmsprop(lr=0.001, decay=1e-6),
#              metrics=['accuracy'])
#'''


model_info = model.fit(X_train, Y_train,
                batch_size = batch_size,
                epochs = nb_epoch,
                verbose = 1,
                validation_data = (X_test, Y_test))

#'''
#model_info = model.fit(X_train, Y_train,
#                batch_size = batch_size,
#                epochs = nb_epoch,
#                verbose=1,
#                validation_split= 0.2)
#'''

#score = model.evaluate(X_test, Y_test, verbose=0)
#print('\nTest loss:', score[0])
#print('Test accuracy:', score[1])
#print(model.predict_classes(X_test[1:5]))
#print(Y_test[1:5])

# -------------------------------------------
# Visualiando as camadas intermediárias

model.summary()

model.get_config()

model.layers[0].input_shape

model.layers[0].output_shape

model.layers[0].get_weights()

np.shape(model.layers[0].get_weights()[0])

model.layers[0].trainable

#np.shape(a)



# with a Sequential model
get_1rd_layer_output = k.function([model.layers[1].input, k.learning_phase()],
                                  [model.layers[0].output])

# output in test mode = 0
#layer_output = get_1rd_layer_output([x, 1])[0]

# output in train mode = 1
#layer_output = get_3rd_layer_output([x, 1])[0]

# A imagem de entrada

#input_image = X_train[0:1,:,:,:]
#print(input_image.shape)

#plt.imshow(input_image[0,0,:,:],cmap ='gray')
#plt.imshow(input_image[0,0,:,:])


#output_image = get_1rd_layer_output(input_image)
#print(output_image.shape)

# -------------------------------------------

# visualizing losses and accuracy

train_loss = model_info.history['loss']
val_loss = model_info.history['val_loss']
train_acc = model_info.history['acc']
val_acc = model_info.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available  # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig("figura1.jpg", dpi=300)

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.show()
# Para rodar no servidor plot com
#plt.savefig("figura2.jpg", dpi=300)



fim = time.time()

print ('\nO Modelo levou {} segundos para treinar a rede ',(fim - inicio))

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # Resumo do historico de precisao
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Modelo de Precisao')
    axs[0].set_ylabel('Precisao')
    axs[0].set_xlabel('Epocas')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # Resumo do historico de Perda
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Modelo de Perda')
    axs[1].set_ylabel('Perda')
    axs[1].set_xlabel('Epocas')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
plt.show()
# Para rodar no servidor plot com
#plt.savefig("Figura3.jpg", dpi=300)

plot_model_history(model_info)

def accuracy(X_test, Y_test, model):
    result = model.predict(X_test)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(Y_test, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

print ('A precisao nos dados de teste e de : ',accuracy(X_test, Y_test, model))


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Exibir Imagens Originais
    ax = plt.subplot(2, n, i+1)
    plt.imshow(X_train[i].reshape(img_rows, img_cols, 3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Exibir Imagens Reconstruidas
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(X_test[i].reshape(img_rows, img_cols, 3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#plt.show()
# Para rodar no servidor plot com
# plt.savefig("Figura4.jpg", dpi=300)

# -------------------------------------------
# Matriz de Confusão
from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print('\nY_pred_X_teste')
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print('\ny_pred_Y_pred')
print(y_pred)

#x_pred = model.predict_classes(X_test)
#print(x_pred)

p=model.predict_proba(X_test) # para prever a probabilidade de seus

target_names = ['  classe 0(  Garca  )', '  classe 1(  Irire  )', 'classe 2(Socozinho)']
print(classification_report(np.argmax(Y_test, axis=1), y_pred, target_names=target_names))
print('\nConfusion Matrix')
print(confusion_matrix(np.argmax(Y_test, axis=1), y_pred))
# -------------------------------------------

# Salvando os pessos
# -------------------------------------------
fname = "Weights.hdf5"
model.save_weights(fname,overwrite=True)
# -------------------------------------------

# Loading weights
# -------------------------------------------
fname = "Weights.hdf5"
model.load_weights(fname)
# -------------------------------------------

