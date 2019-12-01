from model import *
from model import unet
from data import *
from keras.callbacks import ModelCheckpoint

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


my_generator= trainGenerator(2, 'C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/BRATS_2015/train','image','label',data_gen_args, save_to_dir= None)

model= unet(input_size=(256,256,1),  loss_function='binary_crossentropy', opt=Adam(lr = 3e-5))
model_checkpoint = ModelCheckpoint('C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(my_generator,steps_per_epoch=300,epochs=5,callbacks=[model_checkpoint])

testGene = testGenerator("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/BRATS_2015/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg",results)


