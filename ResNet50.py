# import necessary packages
import keras.applications.resnet as resnet
import keras
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.applications import vgg16
from keras.applications import vgg19
import keras.applications.resnet as resnet
from keras.applications.resnet50 import ResNet50
from keras.applications import inception_resnet_v2
from keras.callbacks.callbacks import ModelCheckpoint
from keras.utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from covid_model import CovidNet
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import argparse
from plot import plot


# initial parameters
data_path = './data1'
model_name = 'resnet50'
epochs = 200
lr = 1e-3
batch_size = 32
img_dims = (256,256,3)
num_classes = 3

print('data_path:',data_path,' model:',model_name,' epochs:',epochs,
    ' learning rate:',lr,' batch size:',batch_size,' input dimension:',img_dims)


#labels: covid:0,normal:1,pneumonia:2
data = []
labels = []
covid_count=0
normal_count=0
pneu_conut = 0
print('loading images from train data:')
image_files = [f for f in glob.glob(data_path+'/train' + "/**/*", recursive=True) if not os.path.isdir(f)] 
random.seed(42)
random.shuffle(image_files)

# create groud-truth label from the image path
for img in image_files[0:100]:

    image = cv2.imread(img)
    
    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]
    if num_classes == 3:
        if label == "covid":
            label = 0
        elif label == "normal":
            label = 1
        else:
            label = 2
    elif num_classes ==2:
        if label == "covid":
            label = 0
        else:
            label = 1     
        
    labels.append([label])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, valX, trainY, valY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)
trainY = to_categorical(trainY, num_classes=num_classes)
valY = to_categorical(valY, num_classes=num_classes)

print('trainX.shape:',trainX.shape,' valX.shape:',valX.shape)

# augmenting datset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
print("loading the model")
#set the model      
model=''                   
if model_name=='inception':
    model = inception_resnet_v2.InceptionResNetV2(include_top=False,weights='imagenet',input_shape=img_dims)
elif model_name in ['vgg16','VGG16']:
    model = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=img_dims)
elif model_name in ['vgg19','VGG19']:
    model = vgg19.VGG19(include_top=False, weights='imagenet',input_shape=img_dims)
elif model_name in ['resnet50','ResNet50']:
    model = ResNet50(include_top=False, weights='imagenet',input_shape=img_dims)
elif model_name in ['resnet101','ResNet101']:
    model = resnet.ResNet101(include_top=False, weights='imagenet',input_shape=img_dims)
# print(model.summary())


#fine tuning
output = model.layers[-1].output
output = keras.layers.Flatten()(output)
i=0
new_model = Model(model.input, output=output)
for layer in new_model.layers:
    layer.trainable = True

model = Sequential()
model.add(new_model)
model.add(Dense(img_dims[0], activation='relu', input_dim=img_dims))
model.add(Dropout(0.3))
model.add(Dense(img_dims[0], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
# model.summary()

#path for saving checpoints:
filepath="./check_points/" + str(num_classes) + 'class_'+model_name + "-{epoch:02d}-{loss:.4f}.h5"
#saving the checkpoints:
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(valX,valY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1,
                        callbacks=callbacks_list)

#if you want saving last model without checkpoints comment top block and uncomment this block:
# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
#                         validation_data=(valX,valY),
#                         steps_per_epoch=len(trainX) // batch_size,
#                         epochs=epochs, verbose=1)
# model.save('./check_points/' + model_name+'.model')

plot(H,epochs,model_name)

