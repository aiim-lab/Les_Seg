from model import unet
from model import *
from data import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


my_generator= trainGenerator(2, 'C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/covera_health/data/train','image','label',data_gen_args, save_to_dir= None)

model= unet(input_size=((img_depth, img_rows, img_cols, 1)),  loss_function='binary_crossentropy', opt=Adam(lr = 3e-5))
# model_checkpoint = ModelCheckpoint('C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg', monitor='loss',verbose=1, save_best_only=True)
# model.fit_generator(my_generator,steps_per_epoch=300,epochs=2,callbacks=[model_checkpoint])
model.fit_generator(my_generator,steps_per_epoch=300,epochs=2)

testGene = testGenerator("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/covera_health/data/validation")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("C:/Users/MANPREET .LAPTOP-U55FJOMD/Desktop/Les_seg",results)


