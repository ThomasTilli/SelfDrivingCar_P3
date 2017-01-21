
import csv
import os
import numpy as np
import cv2
from scipy import misc
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, Flatten, Reshape, ELU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


#loading images and preprocessing
datafolders=["d1/","d2/","data/"]
#Load one line from the csv files
#load the corresponding images
#cropp the images on top to remove sky and on bottom to remove car [60:140,0:320] -> new image size [80,320]
#resize the images to [48,48]
#for left and right camera images adjust the steering angle by some small amount


def load_image(fname,folder):
    fname=fname.replace('D:\\selfdrivingcar\\','') #simulator for windows added drive and absolute path
    fname=fname.replace('\\','/').strip()
    #hack for Udacity data set
    if folder=="data/":
        fname=folder+fname   
    img = cv2.imread(fname)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   
    return(img)   

new_size_col,new_size_row = 48, 48
def preprocess_image(img):
    shape = img.shape
    img = img[60:140, 0:320]
    img = cv2.resize(img,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)
    #normalize
    img = img/255.0 - 0.5
    return img

# some data augmentation operations to increase the data set and make it more general
#source https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.cuio5w848
#change brightness, horizontal and vertical shifts, add random shadows

def augment_brightness_camera_images(image):

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

#horizontal and vertical shifts
def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1], image.shape[0]))
    
    return image_tr,steer_ang

#shadow augmentation
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def augment_one_image(img,steer):
    #randomly select left center or right camera
   
    img,steer = trans_image(img,steer,50)
    img = augment_brightness_camera_images(img)
    img=add_random_shadow(img)
    img=preprocess_image(img)
    img = np.array(img)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        img = cv2.flip(img,1)
        steer = -steer
    return img,steer


#load data and augment
#steering angle adjust (rough estimate)
ST_ADJUST=0.25
def load_and_process_data(datafolders):
    images=[]
    steerings = []
    for folder in datafolders:
    
        with open(folder+'driving_log.csv', 'r') as f:
            reader = csv.reader(f)
            i=0
            for row in reader:
                i+=1
                steerC=float(row[3])
                #Center image
                imgCraw=load_image(row[0],folder)
                imgC=preprocess_image(imgCraw)
                images.append(imgC)
                steerings.append(steerC)
               
                #left image
                imgLraw=load_image(row[1],folder)
                imgL=preprocess_image(imgLraw)
              
                images.append(imgL)
                steerL=steerC+ST_ADJUST
                steerings.append(steerL)
                
                #right image
                imgRraw=load_image(row[2],folder)
                imgR=preprocess_image(imgRraw)
                images.append(imgR)
                steerR=steerC-ST_ADJUST
                steerings.append(steerR)
                
            
                #flip center image
                images.append(cv2.flip(imgC, 1))
                steerings.append(steerC*-1.0)
              
                #flip left image
                images.append(cv2.flip(imgL, 1))               
                steerings.append(steerL*-1.0)
                
                #flip right image               
                images.append(cv2.flip(imgR, 1))               
                steerings.append(steerR*-1.0)
                
                #do some data augmentation
                for n in range(0,1):
                    
                    #center images
                    imgC,steerC= augment_one_image(imgCraw,steerC)
                    images.append(imgC)
                    steerings.append(steerC)
                    #left images
                    imgL,steerL= augment_one_image(imgLraw,steerL)
                    images.append(imgL)
                    steerings.append(steerL)
                    #right images
                    imgR,steerR= augment_one_image(imgRraw,steerR)
                    images.append(imgR)
                    steerings.append(steerR)
                
            print(i," rows loaded from: ",folder)    
    npImgs = np.asarray(images)  
    del images
    npSteer=np.asarray(steerings)
    del steerings
    return  npImgs,npSteer

def train_model():
    images,steerings=load_and_process_data(datafolders)
    print(len(images), "images")
    print(len(steerings), "steerings")
    #shuffle data sets and create training and validating data set

    X_tr, X_val, Y_tr, Y_val = train_test_split(images, steerings, test_size=0.1, random_state=1)
    print("done")
    
    #build model
    datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0)    
    from keras.layers.normalization import BatchNormalization
    dout=0.5
    model = Sequential()
    
    model.add(Convolution2D(32, 7, 7, border_mode='same',
                            input_shape=X_tr[0].shape))
    model.add(Activation('relu'))
    
    
    model.add(Convolution2D(64, 7, 7))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Convolution2D(64, 7, 7, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dout))
    
    model.add(Flatten())
    model.add(Dropout(dout))
    model.add(Dense(256))
    model.add(Activation('elu'))
    model.add(Dropout(dout))
    
    model.add(Dense(64))
    model.add(Activation('elu'))
    model.add(Dropout(dout))
    
    model.add(Dense(1))
    model.summary()
    
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['mse'])    
    
    from keras.callbacks import ModelCheckpoint
   
    model.fit_generator(generator=datagen.flow(X_tr, Y_tr, batch_size=128),
              samples_per_epoch=X_tr.shape[0],
              nb_epoch=12,
              validation_data=(X_val, Y_val),
              callbacks=[ModelCheckpoint('cnn_model.h5',save_best_only=True)])    
    
    
    #Saving the model and the weights
    import json
    from keras.models import model_from_json
    
    data = model.to_json()
    
    with open('model.json', 'w') as outfile:
        outfile.write(data)
    
    model.save_weights('model.h5')
    print('Model saved!')    
    
    
if __name__ == '__main__':
    train_model()    