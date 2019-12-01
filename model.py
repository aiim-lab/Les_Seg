import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from losses import *
from keras.callbacks import ModelCheckpoint


def unet(input_size, loss_function, opt):
    filter_size=3
    inputs=Input(shape=input_size)
    conv1= Conv2D(64, filter_size,activation='relu', padding='same')(inputs)
    conv1= Conv2D(64, filter_size, activation='relu', padding='same')(conv1)
    pool1= MaxPooling2D(pool_size=(2,2))(conv1)
    #downsample 1st level
    conv2=Conv2D(128, filter_size, activation='relu', padding='same')(pool1)
    conv2=Conv2D(128, filter_size, activation='relu', padding='same')(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    #downsample 2nd level
    conv3=Conv2D(256, filter_size, activation='relu', padding='same')(pool2)
    conv3=Conv2D(256, filter_size, activation='relu', padding='same')(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(conv3)
    #downsample 3rd level
    conv4=Conv2D(512, filter_size, activation='relu', padding='same')(pool3)
    conv4=Conv2D(512, filter_size, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4=MaxPooling2D(pool_size=(2,2))(drop4)
    #downsample 4th level
    conv5=Conv2D(1024, filter_size, activation='relu', padding='same')(pool4)
    conv5=Conv2D(1024, filter_size, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # pool5=MaxPooling2D(pool_size=(2,2))(conv5)

    #upsampling 4th level
    ups= UpSampling2D(size=(2,2))(drop5)
    conv= Conv2D(512, 2, activation='relu',padding='same')(ups)
    skip= concatenate([conv4,conv], axis=3)
    conv=Conv2D(512, filter_size, activation='relu', padding='same')(skip)
    conv=Conv2D(512, filter_size, activation='relu', padding='same')(conv)

    #upsampling 3rd level
    ups= UpSampling2D(size=(2,2))(conv)
    conv= Conv2D(256, 2, activation='relu',padding='same')(ups)
    skip= concatenate([conv,conv3], axis=3)
    conv=Conv2D(256, filter_size, activation='relu', padding='same')(skip)
    conv=Conv2D(256, filter_size, activation='relu', padding='same')(conv)

    #upsampling 2nd level
    ups= UpSampling2D(size=(2,2))(conv)
    conv= Conv2D(128, 2, activation='relu',padding='same')(ups)
    skip= concatenate([conv,conv2], axis=3)
    conv=Conv2D(128, filter_size, activation='relu', padding='same')(skip)
    conv=Conv2D(128, filter_size, activation='relu', padding='same')(conv)

    #upsampling 1st level
    ups= UpSampling2D(size=(2,2))(conv)
    conv= Conv2D(64, 2, activation='relu',padding='same')(ups)
    skip= concatenate([conv,conv1], axis=3)
    conv=Conv2D(64, filter_size, activation='relu', padding='same')(skip)
    conv=Conv2D(64, filter_size, activation='relu', padding='same')(conv)
    conv= Conv2D(1,(1,1), activation= 'sigmoid', name='final')(conv)

    model= Model(input=inputs, output=conv)
    model.compile(optimizer= opt, loss=loss_function , metrics=[dsc,tp,tn])

    return model













