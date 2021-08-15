#(1)Preparation
import csv
import cv2
import math
import random
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
       
#with open('data2/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)
        
with open('data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#with open('data4/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)
        
#with open('data5/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)

with open('data6/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
with open('data7/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open('data9/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open('data10/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# To use "model.fit_generator" instead of "model.fit", split training data here
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#images = []
#measurements = []
#for line in lines:
#    source_path = line[0]
#    filename = source_path.split('/')[-1]
#    if source_path.split('/')[2] == 'drive1':
#        current_path = 'data1/IMG/' + filename
#    elif source_path.split('/')[2] == 'drive2':
#        current_path = 'data2/IMG/' + filename
#    elif source_path.split('/')[2] == 'drive3':
#        current_path = 'data3/IMG/' + filename
#    elif source_path.split('/')[2] == 'drive4':
#        current_path = 'data4/IMG/' + filename
#    elif source_path.split('/')[2] == 'drive5':
#        current_path = 'data5/IMG/' + filename
#    elif source_path.split('/')[2] == 'drive6':
#        current_path = 'data6/IMG/' + filename
#        
#    image = cv2.imread(current_path)
#    images.append(image)
#    measurement = float(line[3])
#    measurements.append(measurement)

print("Preparation is done.")    
    
#(2)Augmentate Images (Flip the image and steering right/left)
#augmented_images, augmented_measurements = [],[]
#for image, measurement in zip(images, measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#    augmented_measurements.append(measurement * -1.0)
#    
#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)

# To use "model.fit_generator" instead of "model.fit", create generator
num_lines = len(lines)
batchsize = 32

def generator(lines, batch_size=batchsize):
    num_lines = len(lines)
    while 1: 
        random.shuffle(lines)
        
        for offset in range(0, num_lines, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            augmented_images = []
            augmented_measurements = []
            
            for batch_sample in batch_samples:
                    source_path = batch_sample[0]
                    filename = source_path.split('IMG')[-1]
                    if source_path.split('/')[2] == 'drive1':
                        current_path = 'data1/IMG/' + filename
                    elif source_path.split('/')[2] == 'drive2':
                        current_path = 'data2/IMG/' + filename
                    elif source_path.split('/')[2] == 'drive3':
                        current_path = 'data3/IMG/' + filename
                    elif source_path.split('/')[2] == 'drive4':
                        current_path = 'data4/IMG/' + filename
                    elif source_path.split('/')[2] == 'drive5':
                        current_path = 'data5/IMG/' + filename
                    elif source_path.split('/')[2] == 'drive6':
                        current_path = 'data6/IMG/' + filename
                    elif source_path.split('/')[2] == 'drive7':
                        current_path = 'data7/IMG/' + filename
                    elif source_path.split('/')[2] == 'drive9':
                        current_path = 'data9/IMG/' + filename
                    elif source_path.split('/')[2] == 'drive10':
                        current_path = 'data10/IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    measurement = float(batch_sample[3])
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(cv2.flip(image,1)) # Addition the flipped image
                    augmented_measurements.append(measurement* -1.0) # Addition the flipped steering angle
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements) 
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batchsize)
validation_generator = generator(validation_samples, batch_size=batchsize)

print("Augmentation and creating generator is done.")

#(3)Make Model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

model = Sequential()

# Normalization by Lambda
model.add(Lambda(lambda x:x/255.0 -0.5, input_shape = (160,320,3)))

# Crop the image to ignore unnecessary area
model.add(Cropping2D(cropping=((70,25),(0,0)))) 

# Architecture 1: NVIDIA Model for autonomous vehicle (section 15)
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.50)) #Add dropout to avoid overfitting
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#num_epoch = 15
num_epoch = 10
#num_epoch = 5
#num_epoch = 3
#num_epoch = 1

# Start Learning and make Model by "model.fit"
#model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = num_epoch)
#model.save('model1.h5')

# Start Learning and make Model by "model.fit_generator"
LEARNING_RATE =0.0001
model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))
history_object = model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_samples)/batchsize),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_samples)/batchsize),
            epochs=num_epoch, verbose=1)

model.save('model.h5')
print("Model is made.")

#plot the graph for training rate and valiation rate
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

print("Show the graph.")

exit()