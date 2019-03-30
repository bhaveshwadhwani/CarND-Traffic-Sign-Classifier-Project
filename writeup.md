# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bar_chart.png "Bar_Chart_train"
[image2]: ./examples/some_examples.png "some_examples"
[image3]: ./examples/normalized.png "Normalized Image"
[image4]: ./examples/Grayscale.png "Grayscale Image"
[image5]: ./examples/augmented_data.png "augmented_data"
[image6]: ./examples/warped.png "Warped image"
[image7]: ./examples/test_images.png "Test images from internet"
[image8]: ./examples/test_images_pics.png "Test images with softmax"
[image9]: ./examples/feature_maps.png "Feature Maps"
[image10]: ./examples/feature_maps_part2.png "Feature_Maps_Layer_2"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bhaveshwadhwani/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:
Orignal data set
---
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

After processing and adding augmented data
---

* The size of training set is 46480(This size is after adding augmented data)
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 1)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among classes .

![Bar_Chart_train][image1]

Here are few sample images from our dataset .

![Some Examples][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale as seen this had worked well for Sermanet and LeCun as described in their traffic sign classification article. It also helps to reduce training time, which was nice when a GPU wasn't available. Some of attributes could not be classified in normal images for which grayscale images helped there .

Here is an example of a traffic sign image before and after grayscaling.

![Grayscale Image][image4]

As a last step, I normalized the image data because the data has mean zero and equal variance . For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data .

![Normalized image][image3]

I decided to generate additional  data as the results were not that good because the classes were not equally balanced , so i needed to add some data to classes which had low training data and balance classes.

To add more data to the the data set, I used the following techniques because four types of method to augment data viz.random_translate, random_scale, random_warp, and random_brightness

Here are example of an original images and an augmented images:

Brightness adjusted :

![Brightness changed][image5]

Warped Image:

![Warped image][image6]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution        	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Input = 28x28x6. Output = 14x14x6 |
| Convolution			| Output = 10x10x16      						|
| RELU					|												|
| Max pooling			| Input = 10x10x16. Output = 5x5x16             |
| Convolution			| Output = 1x1x16   							|
| RELU					| 												|
| Flatten        		| Input = 5x5x16. Output = 400.                 |
| Flatten        		| Input = 1x1x400. Output = 400 				|
| Concat Flat Layers  	| Input = 400 + 400. Output = 800    			|
| Dropout layer			|												|
| Fully Connected		| Input = 800. Output = 43						|


### Architecture
**Layer 1: Convolutional.** The output shape should be 28x28x6.

**RELU ** 

**Pooling.** The output shape should be 14x14x6.

**Layer 2: Convolutional.** The output shape should be 10x10x16.

**RELU **

**Pooling.** The output shape should be 5x5x16.

**Layer 3: Convolutional.** The output shape should be 1x1x400.

**RELU **

**Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.

**Layer 4: Fully Connected (Logits).** This should have 43 outputs.

### Output
Return the result of the fully connected layer.

#### Output shown in cell output under heading Training Pipeline 
            layer 1 shape: (?, 28, 28, 6)
            Pooling layer: (?, 14, 14, 6)
            layer 2 shape: (?, 10, 10, 16)
    After pooling layer 2: (?, 5, 5, 16)
            layer 3 shape: (?, 1, 1, 400)
        layer2flat shape : (?, 400)
            xflat shape  : (?, 400)
                x shape  : (?, 800)
            Final Logits : (?, 43)


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Multi layer Convulutional nueral network with kernel size 5x5 and stride 1x1 with `VALID` Padding . Learning rate = 0.0009 , epoch = 60 , batch size = 100. I tried with starting with parameters values  like Learning rate = 0.001 , epoch = 20 , batch size = 128 as we had done in LeNet Labs but accuracy was between 85-90 % . Tried playing around with parameters found above ones as good results . While i had got results in epoch = 20 as 95 % in training with higher epoch i got better results and before data augmentations the results were not that good as the classes were in equally balanced , so i needed to add some data to classes which had low training data and balance classes.In addition to this i hadve used RELU and dropout layers for better results .Last but not the least for Optimization i have used ADAM optimizer which is really popular for optimizing on such tasks



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 97.1
* test set accuracy of 94.7

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    Basic Lenet architecture with 2 convolutional neural network layers and one 2 fully connected layers 
* What were some problems with the initial architecture?
    Lower accuracy as less no of layers were present . 
* How was the architecture adjusted and why was it adjusted? 
    Lowered the learning rate .Increased the epoch and reduced the batch size . Included Dropout layer as a part of regularization technique . Previous architecture was over fitting on data hence we were getting lower accuracy on validation set.


* Which parameters were tuned? How were they adjusted and why?
    Learning rate , epoch , batch_size , No. of convulation layers .Above mentioned parameters were tried and tested on validation with various values and best resulting values were kept in final model to achieve best results 
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    Choosing the perfect kernel,filter,strides for Covolutional Nueral Network are some of important decision choices as they can help getting desired results out of input images.Dropouts can help in reducing the dependencies on activation outputs . Our model gets less dependent on input from activation outputs and keeps redundant values to learn for better predictions .
    
If a well known architecture was chosen:
* What architecture was chosen?
    First the basic LeNet architecture was used in solving this problem .

* Why did you believe it would be relevant to the traffic sign application?
    As it was used earlier for similar problem and we had used on MNIST data in LeNet Lab session so it was a good kickstart for this project .

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
     For training set we got good accuracy and on validation set we got good accuracy even though we had less no. of data points in validation it was set to good accuracy .Our model performed well on test set and test images from internet also got good accuracy of nearly 100 % . Only one image was having 95% that too was not a bad one .By checking and analyzing result we can say the model did well on this part .

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![Test images from internet][image7]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![Test images with softmax][image8]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Input Image label  (12, b'Priority road')
----------------------------------------
    1st Guess :Priority road(12)
    100.%

    2nd Guess :Roundabout mandatory(40)
    4.73%

    3rd Guess :Turn left ahead(34)
    3.57%

    4th Guess :Turn left ahead(34)
    5.15%

    5th Guess :Turn left ahead(34)
    1.78%

----------------------------------------

Input Image label  (18, b'General caution')
----------------------------------------
    1st Guess :General caution(18)
    100.%

    2nd Guess :Pedestrians(27)
    1.05%

    3rd Guess :Traffic signals(26)
    3.83%

    4th Guess :Traffic signals(26)
    1.98%

    5th Guess :Traffic signals(26)
    0.0%

----------------------------------------

Input Image label  (34, b'Turn left ahead')
----------------------------------------
    1st Guess :Turn left ahead(34)
    100.%

    2nd Guess :Keep right(38)
    5.50%

    3rd Guess :No vehicles(15)
    4.81%

    4th Guess :No vehicles(15)
    1.16%

    5th Guess :No vehicles(15)
    2.79%

----------------------------------------
Input Image label  (11, b'Right-of-way at the next intersection')
----------------------------------------
    1st Guess :Right-of-way at the next intersection(11)
    100.%

    2nd Guess :Beware of ice/snow(30)
    1.14%

    3rd Guess :Double curve(21)
    6.98%

    4th Guess :Double curve(21)
    1.56%

    5th Guess :Double curve(21)
    5.58%

----------------------------------------
Input Image label  (38, b'Keep right')
----------------------------------------
    1st Guess :Keep right(38)
    100.%

    2nd Guess :Turn left ahead(34)
    2.24%

    3rd Guess :Speed limit (20km/h)(0)
    0.0%

    4th Guess :Speed limit (20km/h)(0)
    0.0%

    5th Guess :Speed limit (20km/h)(0)
    0.0%

----------------------------------------
Input Image label  (1, b'Speed limit (30km/h)')
----------------------------------------
    1st Guess :Speed limit (30km/h)(1)
    98.5%

    2nd Guess :Speed limit (20km/h)(0)
    1.43%

    3rd Guess :Speed limit (80km/h)(5)
    1.14%

    4th Guess :Speed limit (80km/h)(5)
    6.76%

    5th Guess :Speed limit (80km/h)(5)
    1.87%

----------------------------------------

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is output from feature maps . The code for this is written in ipython notebook cell .

First convolutional layer

![Feature Maps][image9]

Second convolutional layer

![Feature_Maps_Layer2][image10]

This is image for Speed limit (120km/h) . We can clearly see in first layer the circle is getting recognised and 120 is very reccognised .
While layer 2 focuses on edges and pixels of 1 direction in particularly . By this we can understand how the nueral network is working by collecting iformation from different structures and parts of image .

Thank you for reading report . Please let me know if any suggestion on above work :)


