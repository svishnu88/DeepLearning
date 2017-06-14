# **Traffic Sign Recognition** 

**In this project we classify traffic signals into one of 43 categories. I have tried using a small architecture that resembles resnet and multi scaling architecture as presented in the [paper](http://yann.lecun.org/exdb/publis/psgz/sermanet-ijcnn-11.ps.gz). The final model achives a validation score of 99.4% . When tested on the data set downloaded from the internet , it correctly classifies 3/5 images. Out of the 5 images , model was not shown 1 category of image , so it predicts a category that looks closer to it.**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Data set exploration and summary 

** Summary stats of the data **
* Training set size is 34799
* Validation set size is 4410 
* Test set size is 12630
* Shape of traffic sign image is (32, 32, 3)
* No of unique classes/labels in the data set is 43
[//]: # (Image References)

[image1]: ./writeupimgs/training_set_distribution.png "TrainingSetDistribution"
[image2]: ./writeupimgs/validation_set_distribution.png "ValidationSetDistribution"
Traing set distribution. It shows that , we have an unbalanced data set.
![alt text][image1]
Validation set distribution . Validation data also has a similar distribution to the training set.
![alt text][image2]


## Preprocessing data
[//]: # (Image References) 

[image3]: ./writeupimgs/trainimagesample.png "Trainig Images"
[image4]: ./writeupimgs/trainaugsample.png "Augmented Images"

* Data is normalized by subtracting each pixel value from 128 and dividing by 128. 
* Since we have a limited data , we augment by applying different augmentation techniques like
    * Random rotation
    * Random zoom
    * Adjusting the range of each channel value.
    * Horizontal flip cannot be used , as some symbols change the meaning of it like left traffic symbol becomes right.I accidentally tried and my validation accuracy did not improve above 93%.
* Sample training data.
![alt text][image3]
* Sample Augmented data.
![alt text][image4]
* Augmenting data gives the deep learning model more data to learn from and prevents overfitting. Here I have used Kera Image generatorn which provides real time augmentation.

## Model Architecture

I tried 2 different architectures .
* Resembles a resnet. I used 4 residual blocks. It was able to achive a training accuracy of 99% and validation accuracy of 97.78 % . I trained the model for 20 epochs . Reduced the learning rate by 10 for every 10 epochs. Though the model could have given a better accuracy if trained for more epochs ,I avoided it as it was taking more time than a simple architecture for each epoch.
* The second architecture I tried was inspired by "Traffic sign recognition with multi-scale convolutional networks". As described in the paper I used a skip connection and allowed the convolutional layer to have different perspective areas. In the original paper, they used a gray scale image, which I did not. Since I observered a color image achives similar result by trying different augmentation techniques.This model achived a training accuracy of 99.37% and validation accuracy of 99.43%.This model was trained for 100 epochs , and reducing the learning rate by 10 for every 20 epochs.
* The first model achieves a test result accuracy of 96.45%
* The model with multiscale convolutional networks result in a test accuracy of 98.5% accuracy.


## Test model on new images
[//]: # (Image References) 

[image5]: ./writeupimgs/german_traffic_test.png "Test Images"
[image6]: ./writeupimgs/german_test_predictions.png "Predicted Images"


* Models will be tested on the below images.
![alt text][image5]
* Model predictions 
![alt text][image6]

The model fails to predict properly for the last 2 images. It some how is confused for 120 as 100. For the last image it makes sense ,it has never seen similar images during training. 

## Top 5 softmax probabilities

[//]: # (Image References)

[image7]: ./writeupimgs/pred_prob_germantest.png "Probabilities"

![alt text][image7]
