#Load all the important packages

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Set a seed

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

# Load the data

with np.load('training-dataset.npz') as data:
    img = data['x']
    lbl = data['y']
    
# Reshape the data

img = img.reshape(img.shape[0], 28, 28)

# Shift the label so it will start from 0 instead of 1

lbl -=1

# Separate the data into training and validation set

trainX, valX, trainY, valY = train_test_split(img, lbl, test_size=0.20, 
                                              random_state=512, stratify = lbl)

# Reshape dataset to have a single channel

width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
valX = valX.reshape((valX.shape[0], width, height, channels))

# One hot encode target values

trainY = to_categorical(trainY)
valY = to_categorical(valY)

# Confirm scale of pixels

print('Train min=%.3f, max=%.3f' % (trainX.min(), trainX.max()))
print('Validation min=%.3f, max=%.3f' % (valX.min(), valX.max()))

# function to add noise
def add_noise(img, sigma=0.05):
    return np.clip(img + np.random.randn(*img.shape)*sigma, 0.0, 1.0)

#add noise
trainX = add_noise(trainX, .2)
valX = add_noise(valX, .2)
print(trainX.shape)

# Create data generator (1.0/255.0 = 0.003921568627451)

datagen = ImageDataGenerator(rescale=1.0/255.0)

# Prepare an iterators to scale images

train_iterator = datagen.flow(trainX, trainY, batch_size=150)
val_iterator = datagen.flow(valX, valY, batch_size=150)
print('Batches train=%d, test=%d' % (len(train_iterator), len(val_iterator)))

# Prepare an iterators to scale images specifically for prediction

val_generator = datagen.flow(
        valX,
        shuffle = False,
        batch_size=150)

# Confirm the scaling works

batchX, batchY = train_iterator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), 
                                              batchX.max()))

# Baseline

baseline = Sequential()
baseline.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 
                                                                channels)))
baseline.add(MaxPooling2D((2, 2)))
baseline.add(Flatten())
baseline.add(Dense(26, activation='softmax'))

optimizer = Adam(lr=0.001)
baseline.compile(optimizer = optimizer, loss = 'categorical_crossentropy', 
                 metrics = ['accuracy'])

# fit model with generator
history_baseline = baseline.fit(train_iterator, steps_per_epoch=len(train_iterator), 
                                epochs=5)

# evaluate model
_, acc_baseline = baseline.evaluate(val_iterator, steps=len(val_iterator), verbose=2)
print('Test Accuracy: %.3f' % (acc_baseline * 100))

# Model 1

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model1.add(MaxPooling2D((2, 2)))
model1.add(Flatten())
model1.add(Dense(64, activation='relu'))
model1.add(Dense(26, activation='softmax'))

optimizer = Adam(lr=0.001)
model1.compile(optimizer = optimizer, loss = 'categorical_crossentropy', 
               metrics = ['accuracy'])

# Fit model 1 with generator

history_model1 = model1.fit(train_iterator, steps_per_epoch=len(train_iterator), 
                            epochs=20)

# Evaluate model 1

_, acc1 = model1.evaluate(val_iterator, steps=len(val_iterator), verbose=2)
print('Test Accuracy: %.3f' % (acc1 * 100))

# Model 2

model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Flatten())
model2.add(Dense(64, activation='relu'))
model2.add(Dense(26, activation='softmax'))

optimizer = Adam(lr=0.001)
model2.compile(optimizer = optimizer, loss = 'categorical_crossentropy', 
               metrics = ['accuracy'])

# Fit model 2 with generator

history_model2 = model2.fit(train_iterator, steps_per_epoch=len(train_iterator),
                            epochs=20)

# Evaluate model 2

_, acc2 = model2.evaluate(val_iterator, steps=len(val_iterator), verbose=2)
print('Test Accuracy: %.3f' % (acc2 * 100))

# Model 3

model3 = Sequential()
model3.add(Conv2D(32, (3, 3), input_shape=(width, height, channels)))
model3.add(LeakyReLU(alpha=0.05))
model3.add(MaxPooling2D((2, 2)))
model3.add(Conv2D(64, (3, 3)))
model3.add(LeakyReLU(alpha=0.05))
model3.add(MaxPooling2D((2, 2)))
model3.add(Flatten())
model3.add(Dense(64))
model3.add(LeakyReLU(alpha=0.05))
model3.add(Dense(26, activation='softmax'))

optimizer = Adam(lr=0.001)
model3.compile(optimizer = optimizer, loss = 'categorical_crossentropy', 
               metrics = ['accuracy'])

# Fit model 3 with generator

history_model3 = model3.fit(train_iterator, steps_per_epoch=len(train_iterator), 
                            epochs=20)

# Evaluate model 3

_, acc3 = model3.evaluate(val_iterator, steps=len(val_iterator), verbose=2)
print('Test Accuracy: %.3f' % (acc3 * 100))

imgs = np.load('test-dataset.npy')

img_rows = imgs.shape[1]
img_cols = imgs.shape[2]

#channels-first/channels-last, backend-agnostic
if K.image_data_format() == 'channels_first':
    imgs = imgs.reshape(imgs.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    imgs = imgs.reshape(imgs.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

imgs = imgs.astype(np.uint8)

def inside(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 > x2) and (y1 > y2) and (x1 + w1 < x2 + w2) and (y1+h1 < y2 + h2)

def wrap_digit(rect, img_w, img_h):
    x, y, w, h = rect
    x_centre = x + w//2
    y_centre = y + h//2
    if (h > w):
        w = h
        x = x_centre - (w//2)
    else:
        h = w
        y = y_centre - (h//2)
    
    padding = 5
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    
    if x < 0 :
        x = 0
    elif x > img_w:
        x = img_w
    
    if y < 0:
        y = 0
    elif y > img_h:
        y = img_h
    
    if x+w > img_w:
        w = img_w - x
    
    if y+h > img_h:
        h = img_h - y
    
    return x, y, w, h

# Function to run prediction
def predict(ann, sample):
    sample = cv2.resize(sample, (28, 28), interpolation=cv2.INTER_LINEAR)
    sample = sample/225
    sample = sample.reshape(1,28,28,1)
    return ann.predict(sample)

# Function to get the value inside a rectangle
def get_rect(img_arr, thr):
    gray = cv2.GaussianBlur(img_arr, (5,5), cv2.BORDER_DEFAULT)
    ret, thresh = cv2.threshold(img_arr, thr, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    img_h, img_w = gray.shape[:2]
    img_area = img_w * img_h
    for c in contours:
        a = cv2.contourArea(c)
        if a >= 0.98 * img_area or a <= 0.0001 * img_area:
            continue
        r = cv2.boundingRect(c)
        is_inside = False
        for q in rectangles:
            if inside(r, q):
                is_inside = True
                break
        if not is_inside:
            rectangles.append(r)
    rectangles = sorted(rectangles, key = lambda x: x[0] , reverse = False)


    near_index = []
    if len(rectangles) > 5 :
        for i in range(len(rectangles)-1):
            if rectangles[i+1][0] - rectangles[i][0] < 5 :
                near_index.append(i+1)
    
    near_index.sort(reverse = True)
    
    for k in range(len(near_index)) :
        rectangles.remove(rectangles[near_index[k]])
    
    if len(rectangles) - 5 > 0 :
        for i in range(len(rectangles) - 5):
            rectangles.sort(key = lambda x : x[2])
            rectangles.remove(rectangles[0])

    rectangles.sort(key = lambda x : x[0])

    return rectangles, img_h, img_w, thresh

# Function to run the prediction with the best model and create a csv file
def pred_to_csv(imgs):
    list_for_pd = []
    
    for each in imgs:
        rectangles, img_h, img_w, thresh = get_rect(each, 120)
    
    
        if len(rectangles) < 5 : 
            for i in range(5 - len(rectangles)):
                rectangles.append(rectangles[-1])
        
        pred_list = []
        for iter_accuracy in range(5):
            pred_str = '' 
            
            for r in rectangles:
                x, y, w, h = wrap_digit(r, img_w, img_h)
                roi = thresh[y:y+h, x:x+w]
        
                digit_class = np.argmax(predict(model3, roi)[0]) + 1
                
                if digit_class < 10 :
                    pred_str += '0' + str(digit_class)
                else :
                    pred_str += str(digit_class)
            
            pred_list.append(pred_str)
      
        list_for_pd.append(pred_list)
    
    
    df = pd.DataFrame(list_for_pd)
    df.to_csv('group-13-predictions.csv', index = False, header = False)

# Run prediction with the best model and create a csv file
pred_to_csv(imgs)
