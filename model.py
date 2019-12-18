import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from losses import *
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D, AveragePooling3D, ZeroPadding3D, Conv2DTranspose
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from losses import binary_crossentropy
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.applications import resnet



img_rows = 256
img_cols = 256
img_depth = 36
smooth = 1.

def unet(input_size, loss_function, opt):
    filter_size=3
    inputs=Input((img_depth, img_rows, img_cols, 1))
    conv11= Conv3D(32, filter_size,activation='relu', padding='same')(inputs)
    conc11= concatenate([inputs, conv11], axis=4)
    conv12= Conv3D(32, filter_size, activation='relu', padding='same')(conc11)
    conc12= concatenate([inputs, conv12], axis=4)
    pool1= MaxPooling3D(pool_size=(2,2,2))(conc12)

    #downsample 1st level
    conv21=Conv3D(64, filter_size, activation='relu', padding='same')(pool1)
    conc21= concatenate([pool1, conv21], axis=4)
    conv22=Conv3D(64, filter_size, activation='relu', padding='same')(conc21)
    conc22= concatenate([pool1,conv22], axis=4)
    pool2=MaxPooling3D(pool_size=(2,2,2))(conc22)

    #downsample 2nd level
    conv31=Conv3D(128, filter_size, activation='relu', padding='same')(pool2)
    conc31= concatenate([pool2, conv31], axis=4)
    conv32=Conv3D(128, filter_size, activation='relu', padding='same')(conc31)
    conc32= concatenate ([pool2, conv32], axis=4)
    pool3=MaxPooling3D(pool_size=(2,2,2))(conc32)

    #downsample 3rd level
    conv41=Conv3D(256, filter_size, activation='relu', padding='same')(pool3)
    conc41= concatenate([pool3, conv41], axis=4)
    conv42=Conv3D(256, filter_size, activation='relu', padding='same')(conc41)
    conc42= concatenate([pool3, conv42], axis=4)
    pool4=MaxPooling3D(pool_size=(2,2,2))(conc42)

    #downsample 4th level
    conv51=Conv3D(512, filter_size, activation='relu', padding='same')(pool4)
    conc51= concatenate([pool4, conv51], axis=4)
    conv52=Conv3D(512, filter_size, activation='relu', padding='same')(conc51)
    conc52= concatenate([pool4, conv52], axis=4)
    

    #upsampling 4th level
    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
    conv61= Conv3D(256, filter_size , activation='relu',padding='same')(up6)
    conc61= concatenate([up6, conv61], axis=4)
    conv62=Conv3D(256, filter_size, activation='relu', padding='same')(conc61)
    conc62= concatenate([up6, conv62], axis=4)

    #upsampling 3rd level
    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc62), conv32], axis=4)
    conv71 = Conv3D(128, filter_size, activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(128, filter_size, activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=4)

    #upsampling 2nd level
    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conv22], axis=4)
    conv81 = Conv3D(64, filter_size, activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(64, filter_size, activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=4)

    #upsampling 1st level
    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
    conv91 = Conv3D(32, filter_size, activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=4)
    conv92 = Conv3D(32, filter_size, activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc92)

    model= Model(input=[inputs], output=[conv10])
    model.compile(optimizer= opt, loss=loss_function , metrics=[dsc,tp,tn])

    return model













