from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
from skimage import img_as_uint


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

color_dict= np.array([Sky,Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

#create 2 instances with same arguments

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
    'C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/BRATS_2015/train/image',
    class_mode=None,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    shuffle=True,
    seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
    'C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/BRATS_2015/train/label',
    class_mode=None,
    target_size=target_size,
    shuffle=True,
    color_mode='grayscale',
    batch_size=batch_size,
    seed=seed)

    #combine generators into one which yields image and masks
    train_generator= zip(image_generator, mask_generator)
    return train_generator

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img/255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img =  item[:,:,0]
        # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        io.imsave(os.path.join(save_path,"%d_predict.tif"%(i)),img_as_uint(img))
















    













