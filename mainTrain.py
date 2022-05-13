import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix
#import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix , classification_report

                    
image_directory='datasets/'

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]

INPUT_SIZE=64




for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):    
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')  
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))    
        label.append(0)                   

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)


x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

#Reshape = (n, image_width, image_height, n_channel)

#print(x_train.shape)
#print(y_train.shape)

#print(x_test.shape)
#print(y_test.shape)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)



# Model Building
# 64,64,3

model=Sequential()  

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))  
model.add(MaxPooling2D(pool_size=(2,2)))        



model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))                   
model.add(Dense(2))      
model.add(Activation('softmax')) 



model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])



epochs = 5  

model.fit(x_train, y_train, 
batch_size=32,    
epochs=epochs,      
verbose=1,                                      
validation_data=(x_test, y_test),               
shuffle=True)                                   
                                                
                            

score = model.evaluate(x_test, y_test, verbose = 1) 
print('Test accuracy', score[1])

#model_acc = model.evaluate(x_test, y_test)[1]
#print('Test accuracy: {:.3%}'.format(model_acc))

