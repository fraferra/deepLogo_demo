from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten, Lambda
from keras.models import Model
from keras.layers import Input
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D

from keras.models import Sequential, ask_to_proceed_with_overwrite
from keras.layers import Dense
from keras.layers.core import Activation, Dense
from keras.layers import SimpleRNN, LSTM,  GRU,SeparableConvolution2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras.layers.wrappers import TimeDistributed

import keras
from keras.applications import VGG16
from keras.models import Model
from keras.layers.core import Activation, Dense
import tensorflow as tf
import glob

import scipy
import numpy as np
from PIL import Image
import random
import pandas as pd
import random
import subprocess as sp
import sys
import glob
from pytube import YouTube
import pytube
max_frames = 10
number_of_brands = 5


def getModel2( output_dim ):
    ''' 
        * output_dim: the number of classes (int)
        
        * return: compiled model (keras.engine.training.Model)
    '''
    #vgg_model = VGG16( weights='imagenet', include_top=False )
    vgg_model = VGG16(weights='imagenet', include_top=False)

#     for layer in vgg_model.layers:
#         layer.trainable = False

    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    
    predictions = Dense(output_dim ,activation='softmax')(x)
    #predictions = Lambda(lambda x:x/1.5)(predictions)
# this is the model we will train
    model = Model(input=vgg_model.input, output=predictions)
    #model.add(Lambda(lambda x: x / 2.0))

    #Freeze all layers of VGG16 and Compile the model
    #Confirm the model is appropriate

    return model    

def load_model(weights_path, output_dim):

    model = getModel2( output_dim ) 

    for k,layer in model.layers_by_depth.items()[:]:
        layer[0].trainable = False
#     for layer in model.layers:
#         layer.trainable = False
        
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'categorical_crossentropy'])   
    return model


#labels_dict = {'apple': 2, 'cocacola': 0, 'nike': 4, 'pepsi': 1, 'starbucks': 3, "noise":5}
labels_dict = {'apple': 2, 'cocacola': 0, 'nike': 3, 'pepsi': 1, "noise":4}

reversed_dict = dict([(x[1],x[0]) for x in labels_dict.items()])

def get_brand(softmaxes):
    v = np.max(softmaxes.mean(axis=0))
    
    print v
    
    print reversed_dict[np.argmax(softmaxes.mean(axis=0))]
    if v > 0.6:
        return reversed_dict[np.argmax(softmaxes.mean(axis=0))]
    return "Noise"


def get_brand_2(results):
    v1 = np.argmax(results)% number_of_brands
    v2 = np.argmax(results.mean(axis=0))
    print np.max(results)
    if v1 == v2:
        return reversed_dict[v1]
    if v1 == 4 and v2 != 4:
        return reversed_dict[v2]
    if v1 !=4 and v2 ==4:
        return reversed_dict[v1]
    if v1==4 and v2==4:
        return reversed_dict[v1]


def load_rnn_model(weights, cnn):
    model_rnn3=Sequential()

    model_rnn3.add(TimeDistributed(cnn, input_shape=(max_frames,224,224,3)))
#     model_rnn3.add(GRU(output_dim=100,return_sequences=True))
#     model_rnn3.add(GRU(output_dim=50,return_sequences=False))
#     model_rnn3.add(Dropout(.2))
#     model_rnn3.add(Dense(number_of_brands,activation='softmax'))
#     model_rnn3.load_weights(weights)

#     model_rnn3.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy', 'categorical_crossentropy'])


    #model_rnn3.add(GRU(output_dim=200,return_sequences=True))
    model_rnn3.add(GRU(output_dim=100,return_sequences=True))
    model_rnn3.add(GRU(output_dim=50,return_sequences=False))
    model_rnn3.add(Dropout(.2))
    model_rnn3.add(Dense(number_of_brands,activation='softmax'))
    model_rnn3.compile(optimizer='Nadam', loss='categorical_crossentropy',metrics=['accuracy', 'categorical_crossentropy'])



    
    return model_rnn3

class DeepLogo():
    def __init__(self):
        self.trained_model = load_model("weights-improvement-VGG16-LOGO6-01-0.9822.hdf5", 5)
        self.rnn_model = load_rnn_model("RNN2-07-0.4483.hdf5", self.trained_model)
        

    def predict(self, url):
        yt = YouTube(url)
        try:
            video = yt.get('mp4')
        except pytube.exceptions.MultipleObjectsReturned:
            video = yt.get('mp4', '720p')
        
        
        cmd1 = "rm tmp/tmp_1"
        sp.call(cmd1,shell=True)
        video.download('tmp/tmp_1')
        
        filename = "tmp/tmp_1"

        cmd1 = "rm -r tmp/tmp_1_1"
        sp.call(cmd1,shell=True)

        cmd1 = "mkdir tmp/tmp_1_1"
        sp.call(cmd1,shell=True)
        outputpath= "tmp/tmp_1_1/"

        outputfile = outputpath + "f"
        cmd2='ffmpeg -i '+filename+' -r 1 -s 224x224 ' + outputfile + '_%04d.jpg'
        sp.call(cmd2,shell=True)
        sys.stdout.write("Saved it at " + outputfile +"\n")
        
        frames = sorted(glob.glob("tmp/tmp_1_1/*"))[:-15]
        x_mean = 0.39533365588770686
        
        imgs = []
        for frame in frames:
            img = np.asarray(Image.open(frame))/255.
            imgs.append(img)

        imgs = np.array(imgs)
        imgs = imgs - imgs.mean()

        print "predicting CNN..."
        softmaxes = self.trained_model.predict(imgs)
        b1 =  get_brand(softmaxes)
        print b1
        avgs = []
        for i in range(0, len(softmaxes)-max_frames ):
            #avgs.append(softmaxes[i])
            avgs.append(softmaxes[i:i+max_frames].mean(axis=0))

        avgs = np.array(avgs)
        top_idxs = []
        for i in range(number_of_brands):
            #print np.argmax(avgs[:,i])
            top_idxs.append(np.argmax(avgs[:,i]))

        print top_idxs
        sequences = []

        for top_f in top_idxs:
            sequences.append(imgs[top_f:top_f+max_frames])

        sequences = np.array(sequences)
        #return sequences,softmaxes,imgs,frames
        print sequences.shape
        print "predicting RNN..."
        ps = self.rnn_model.predict(sequences)
        
        b2 = get_brand_2(ps)
        if b1 == b2:
            brand =  b1
        else:
            brand = "Noise"
        return b2
