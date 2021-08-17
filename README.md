# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

1.Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, I used what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

I used a simulator where I can steer a car around a track for data collection. I used image data and steering angles to train a neural network and then used this model to drive the car autonomously around the track.

I created a detailed writeup of the project. Please check it for the details.       
[Writeup_of_Lesson17.pdf](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/Writeup_of_Lesson17.pdf) 

I submitted the following five files which are required: 
* [model.py](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/model.py)  (script used to create and train the model) 
* [drive.py](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/drive.py)  (script to drive the car - feel free to modify this file)
* [model.h5](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/model.h5) (a trained Keras model)
* [a report writeup file](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/Writeup_of_Lesson17.pdf) (pdf)
* [video.mp4](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/video.mp4) (a video recording of your vehicle driving autonomously around the track for at least one full lap)

2.The Rublic
---
 [Link of rubric points](https://review.udacity.com/#!/rubrics/432/view) 

### (1)Required Files  
The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4.  
--> They are all submitted as written above.  

### (2)Quality of Code  
* The model provided can be used to successfully operate the simulation.  
 --> It successfully operated the simulation in Udaciry Workspace.
* The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.  
 --> I used generator. I wrote comments to make the code readable.

### (3)Model Architecture and Training Strategy
* The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.  
  --> I added 5 convolution layers. I used RELU as activation function to introduce nonlinearlity. I added normalized layer by using Lambda function.
* Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.  
  --> I splitted Train/validation/test data. I added dropout layer between the convolutional layer and flattend layer.  
* Learning rate parameters are chosen with explanation, or an Adam optimizer is used.  
  --> I used Adam optimizer.
* Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).  
  --> I chose the following Training dataset.  
  * Keeping the car on the track.
  * Drive the curve smoothly.
  * Recovery laps to avoid lane deviation.

### (4)Architecture and Training Documentation
* The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.  
  --> I discussed it in [Writeup_of_Lesson17.pdf](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/Writeup_of_Lesson17.pdf) .
* The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged. Here is one such tool for visualization.  
  --> I provided it in [Writeup_of_Lesson17.pdf](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/Writeup_of_Lesson17.pdf) .
* The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included.  
  --> I described it in [Writeup_of_Lesson17.pdf](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/Writeup_of_Lesson17.pdf) .


### (5)Simulation
* No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).  
  --> The vehicle drove well in the track 1. [video.mp4](https://github.com/kkumazaki/Self-Drivig-Car_Project4_Behavioral-Cloning/blob/master/video.mp4)


3.The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.
