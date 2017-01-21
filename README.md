# Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project

The objective of this project is to develop a Deep Neural Network which can drive a simulated vehicle in a simlated environment. The simulated environment is built on the Unity engine and consits of a single vehicle in a small world.

## Data
I used the dataset provided by Udacity. Then I created a dataset in the training mode of the simulator for track 1 and one for track 2 using the highest resolution and highest quality of the simulator using a joystick. I used the center camera images and the left and right camera images as well. An adjustment of +0.25 has been made to the steering angle for left camera image and added to the training labels, so that if the car sees a scene similar to left camera it can find the appropriate steering angle. Similar to above an adjustment of -0.25 has been made to right camera image.

## Preprocessing
The images were cropped at the bottom to remove the car and at the top to remove the sky resulting in an image size of 320 widht and 80 height. Then the images were normalized by dividing by 255 and then subtracting 0.5.

## Data Augmentation
The three data sets contains 15017 rows in summary, this are 45057 images in summary. Not very much images for a deep learning network at all. Therefore I added some data augmentation described here  https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.n1bqz8ofx:

- Brightness augmentation to simulate day and night conditions.
- Horizontal and vertical shifts to simulate the effect of car being at different positions on the road.
- Shadow augmentation where random shadows are cast across the image.

After data augmentation I had a dataset of 135153 images which fits into 16GB RAM.

## Splitting data set into training and validation data set
I used 10% of the data set as validation data set.  This only gives an indication for overfitting,  the real test can only be done in the simulator.

## Network architecture
My network architecture is partly based on the paper  End to End Learning for Self-Driving Cars (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I used 5 conv layers with RELU activation and batch normalization and then after a flatten layer two fully-connected layers with ELU-Activation (RELU activation turns out does not work as well especially for large steering angles). The detailed network architecture is:

___________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_181 (Convolution2D (None, 48, 48, 32)    4736        convolution2d_input_37[0][0]     
____________________________________________________________________________________________________
activation_260 (Activation)      (None, 48, 48, 32)    0           convolution2d_181[0][0]          
____________________________________________________________________________________________________
convolution2d_182 (Convolution2D (None, 42, 42, 64)    100416      activation_260[0][0]             
____________________________________________________________________________________________________
batchnormalization_129 (BatchNor (None, 42, 42, 64)    256         convolution2d_182[0][0]          
____________________________________________________________________________________________________
activation_261 (Activation)      (None, 42, 42, 64)    0           batchnormalization_129[0][0]     
____________________________________________________________________________________________________
maxpooling2d_109 (MaxPooling2D)  (None, 21, 21, 64)    0           activation_261[0][0]             
____________________________________________________________________________________________________
convolution2d_183 (Convolution2D (None, 21, 21, 64)    200768      maxpooling2d_109[0][0]           
____________________________________________________________________________________________________
batchnormalization_130 (BatchNor (None, 21, 21, 64)    256         convolution2d_183[0][0]          
____________________________________________________________________________________________________
activation_262 (Activation)      (None, 21, 21, 64)    0           batchnormalization_130[0][0]     
____________________________________________________________________________________________________
convolution2d_184 (Convolution2D (None, 21, 21, 64)    102464      activation_262[0][0]             
____________________________________________________________________________________________________
batchnormalization_131 (BatchNor (None, 21, 21, 64)    256         convolution2d_184[0][0]          
____________________________________________________________________________________________________
activation_263 (Activation)      (None, 21, 21, 64)    0           batchnormalization_131[0][0]     
____________________________________________________________________________________________________
maxpooling2d_110 (MaxPooling2D)  (None, 10, 10, 64)    0           activation_263[0][0]             
____________________________________________________________________________________________________
convolution2d_185 (Convolution2D (None, 10, 10, 64)    102464      maxpooling2d_110[0][0]           
____________________________________________________________________________________________________
batchnormalization_132 (BatchNor (None, 10, 10, 64)    256         convolution2d_185[0][0]          
____________________________________________________________________________________________________
activation_264 (Activation)      (None, 10, 10, 64)    0           batchnormalization_132[0][0]     
____________________________________________________________________________________________________
maxpooling2d_111 (MaxPooling2D)  (None, 5, 5, 64)      0           activation_264[0][0]             
____________________________________________________________________________________________________
dropout_134 (Dropout)            (None, 5, 5, 64)      0           maxpooling2d_111[0][0]           
____________________________________________________________________________________________________
flatten_37 (Flatten)             (None, 1600)          0           dropout_134[0][0]                
____________________________________________________________________________________________________
dropout_135 (Dropout)            (None, 1600)          0           flatten_37[0][0]                 
____________________________________________________________________________________________________
dense_112 (Dense)                (None, 256)           409856      dropout_135[0][0]                
____________________________________________________________________________________________________
activation_265 (Activation)      (None, 256)           0           dense_112[0][0]                  
____________________________________________________________________________________________________
dropout_136 (Dropout)            (None, 256)           0           activation_265[0][0]             
____________________________________________________________________________________________________
dense_113 (Dense)                (None, 64)            16448       dropout_136[0][0]                
____________________________________________________________________________________________________
activation_266 (Activation)      (None, 64)            0           dense_113[0][0]                  
____________________________________________________________________________________________________
dropout_137 (Dropout)            (None, 64)            0           activation_266[0][0]             
____________________________________________________________________________________________________
dense_114 (Dense)                (None, 1)             65          dropout_137[0][0]                
====================================================================================================
Total params: 938,241
Trainable params: 937,729
Non-trainable params: 512

About 4 times more parameters as in the paper referenced above. I not tried a simpler network because after a number of trials and errors with number of epochs and network architectures it worked quite well.

## Training
I used ADAM optimizer with a learning parameter of 0.0001. After some experiments with the final model above I get a quite good model after training 12 epochs.

## Results
The resulting model works well with all screen resolutions and graphics qualities on track 1. On track 2 it works well with all screen resolutions and almost all graphcis qualities but with simple and good graphics qualities it fails because the shadow handling is quite different. More data augmentation and training with this graphics qualities would possible solve this problem.  This shows that data augmentation maybe is not enough, much training data on several tracks with different lighting and weather conditions are required to get a model which works well on almost every track with all possible conditions.

The following movies shows the car driving on both tracks with a screen resolution of 1280x960 and graphics quality "fantastic":

https://youtu.be/dj0LpuAZ4zs

https://youtu.be/D9A6UuvWOlM


