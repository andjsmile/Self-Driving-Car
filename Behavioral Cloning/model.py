import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import randint
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard

#####################################################################################
# lines=[]
# path='/home/andjsmile/anaconda3/udacity/CarND-Behavioral-Cloning-P3/data/data/'
# images=[]
# measurements=[]
# num=0
# with open('./data/data/driving_log.csv') as csvfile:
#     reader=csv.reader(csvfile)
#     for line in reader:
#         if num>0:
#             steering_center=float(line[3])
#             correction=0.15
#             steering_left=steering_center+correction
#             steering_right=steering_center-correction
#             img_center=cv2.imread(path+line[0])
#             img_left=cv2.imread(path+line[1])
#             img_right=cv2.imread(path+line[2])
#             images.extend(img_center,img_left,img_right)
#             measurements.extend(steering_center,steering_left,steering_right)
#     else:
#         num=1
#
# augmented_images,augmented_measurements=[],[]
# for image,measurement in zip(images,measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flip(image,1))
#     augmented_measurements.append(measurement*-1.0)
#
# X_train=np.array(augmented_images)
# y_train=np.array(augmented_measurements)
#######################################################################################
correction=0.10
num=0
samples=[]
with open('./data/data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        if num>0:
            samples.append(line)
        else:
            num=1

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples,batch_size=64):
    num_samples=len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            images=[]
            angles=[]
            for batch_sample in batch_samples:
                path_center='/home/andjsmile/anaconda3/udacity/CarND-Behavioral-Cloning-P3/data/data/IMG/'+ batch_sample[0].split('/')[-1]
                path_left = '/home/andjsmile/anaconda3/udacity/CarND-Behavioral-Cloning-P3/data/data/IMG/' + batch_sample[1].split('/')[-1]
                path_right = '/home/andjsmile/anaconda3/udacity/CarND-Behavioral-Cloning-P3/data/data/IMG/' + batch_sample[2].split('/')[-1]
                image_center=cv2.imread(path_center)
                image_left = cv2.imread(path_left)
                image_right=cv2.imread(path_right)
                images.extend([image_center,image_left,image_right])
                angles.extend([float(batch_sample[3]), float(batch_sample[3]) + correction, float(batch_sample[3]) -correction])


            augmented_images, augmented_measurements = [], []
            for image,measurement in zip(images,angles):
                # if np.random.uniform() > 0.5:
                #     latShift = randint(-5, 5)
                #     M = np.float32([[1, 0, latShift], [0, 1, 0]])
                #     imgTranslated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                #     augmented_images.append(imgTranslated )
                #     augmented_measurements.append(measurement)
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                if measurement>0.15 or measurement<-0.20:
                    augmented_images.append(cv2.flip(image,1))
                    augmented_measurements.append(measurement*-1.0)
            X_train=np.array(augmented_images)
            y_train=np.array(augmented_measurements)
            yield shuffle(X_train,y_train)


# compile and train the model using the generator function
train_generator=generator(train_samples,batch_size=64)
validation_generator=generator(validation_samples,batch_size=64)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
# model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
# model.add(Flatten())
# model.add(Dense(1))
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(filters=24,kernel_size=(5,5),strides=(2,2),kernel_initializer='glorot_uniform', bias_initializer='zeros',activation='relu'))
model.add(Convolution2D(filters=36,kernel_size=(5,5),strides=(2,2),kernel_initializer='glorot_uniform', bias_initializer='zeros',activation='relu'))
model.add(Convolution2D(filters=48,kernel_size=(5,5),strides=(2,2),padding='valid',kernel_initializer='glorot_uniform', bias_initializer='zeros',activation='relu'))
model.add(Convolution2D(filters=64,kernel_size=(3,3),padding='valid',kernel_initializer='glorot_uniform', bias_initializer='zeros',activation='relu'))
model.add(Convolution2D(filters=64,kernel_size=(3,3),padding='valid',kernel_initializer='glorot_uniform', bias_initializer='zeros',activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
#model.fit_generator(train_generator,steps_per_epoch=100,epochs=5,verbose=1)
history=model.fit_generator(generator=train_generator, samples_per_epoch= len(samples)/64,validation_data=validation_generator,nb_val_samples=len(validation_samples)/64,epochs=6,verbose=2,callbacks=[TensorBoard(log_dir='mytensorboard/3')])
model.evaluate_generator(validation_generator,64)
model.save('model.h5')
print('save sucessfully')
model.summary()
