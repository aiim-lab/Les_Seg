from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
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
from PIL import Image
import os
import scipy.misc
from pathlib import Path

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

# #converting training image data to image
# Pathdata_train_image= Path("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg/data/data/data/Training/DomainA/image")
# Pathdata_train_label= Path("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg/data/data/data/Training/DomainA/label")
# Outputpath_image= Path("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg/data/data/data/Training/DomainA/image")
# Outputpath_label= Path("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg/data/data/data/Training/DomainA/label")

# files_train_image=[]
# for dirName, subdirList,fileList in os.walk(Pathdata_train_image):
#     for filename in fileList:
#         if ".npy" in filename.lower(): #check if the file is npy
#             files_train_image.append(os.path.join(dirName,filename))

# #Looping through all the files
# for files in files_train_image:
#     ds= np.load(files, allow_pickle=True)
#     ds_new=Image.fromarray(np.moveaxis(ds,0,-1).save(Outputpath_image/f"{filename}.png"))


# #converting training label data to image
# for filename in os.listdir(Pathdata_train_image):
#     if ".npy" in filename.lower(): #check if the file is npy
#         Image.fromarray(np.moveaxis(filename,0,-1).save(Outputpath_label/f"{filename}.png"))


# #converting test data to images
# Pathdata_train= Path("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg/data/data/data/Training/DomainB")
# Outputpath= Path("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg/data/data/data/Training/DomainB")

# for filename in os.listdir(Pathdata_train):
#     if ".npy" in filename.lower(): #check if the file is npy
#         Image.fromarray(np.moveaxis(filename,0,-1).save(Outputpath/f"{filename}.png"))




#create 2 instances with same arguments

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 2,save_to_dir = None,target_size = (36,256,256),seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
    'C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/covera_health/data/train',
    class_mode=None,
    target_size=target_size,
    color_mode='grayscale',
    batch_size=batch_size,
    shuffle=True,
    seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
    'C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/covera_health/data/train',
    class_mode=None,
    target_size=target_size,
    shuffle=True,
    color_mode='grayscale',
    batch_size=batch_size,
    seed=seed)

    #combine generators into one which yields image and masks
    train_generator= zip(image_generator, mask_generator)
    return train_generator

def testGenerator(test_path,num_image = 4,target_size = (36,256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.npy"%i),as_gray = as_gray)
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
















    













