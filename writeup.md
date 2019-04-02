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
[image7]: ./examples/internet_images.png "Test images from internet"
[image8]: ./examples/images_with_top3.png "Test images with softmax"
[image9]: ./examples/feature_maps.png "Feature Maps"
[image10]: ./examples/feature_maps_part2.png "Feature_Maps_Layer_2"
[image11]: ./examples/chart_softmax.png "Test images with softmax Chart"
[image12]: ./examples/modifiedLeNet.jpeg "Architechture Diagram"
[image13]: ./examples/LeNet_Original_Image.jpg "LeNet_OriginalArchitechture Diagram"
[image14]: ./examples/final_tuned.png "Tuned Model Accuracy plot"

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

Architechture Daigram :
---
Here is diagram of architecture used in this project . It is modified LeNet architecture which is based on LeNet architecture which was designed was Yann LeCun in late 1980's . You can get a breif overview over [here](http://yann.lecun.com/exdb/lenet/) 

![Architechture Diagram][image12]

Here we have original Lenet architecture .

![LeNet_OriginalArchitechture Diagram][image13]

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

To train the model, I used an Multi layer Convulutional nueral network with kernel size 5x5 and stride 1x1 with `VALID` Padding . Learning rate = 0.0001 , epoch = 65 , batch size = 100. I tried with starting with parameters values  like Learning rate = 0.0001 , epoch = 20 , batch size = 128 as we had done in LeNet Labs but accuracy was between 85-90 % . Tried playing around with parameters found above ones as good results . While i had got results in epoch = 20 as 95 % in training with higher epoch i got better results and before data augmentations the results were not that good as the classes were in equally balanced , so i needed to add some data to classes which had low training data and balance classes.In addition to this i hadve used RELU and dropout layers for better results .Last but not the least for Optimization i have used ADAM optimizer which is really popular for optimizing on such tasks. Previously model was overfitting so i had to tune model with parameter so that it should not over fit training data .
I used dropout layer and tuned parameter keep_prob ,learning rate epoch for getting greater generalization on model .
i had reduced learning rate and had checked performance of model on various epoch's to check how well it is doing .

Here is plot of epoch vs training accuracy and validation accuracy .

![Tuned Model Accuracy plot][image14]

I had started with checking accuracy and loss values on various epoch's and stopped where the validation accuracy plateau's . from 60th to 65th epoch we don't have much increase in training as well as validation accuracy . I have iplemented a condition in code as to when validation accuracy plateu's or decreases we need to stop training . This code is written in section "Train Model" . While i had tried previously with manual interaction and expermentation that after 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 96.9
* validation set accuracy of 93.8
* test set accuracy of 92.5

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    Lenet architecture was chosen for this projet as it has been used  previously for similar task of image classificatio giving good results .
* What were some problems with the initial architecture?
    Lower accuracy as less no of layers were present . Model was overfitting to training data  
* How was the architecture adjusted and why was it adjusted? 
    Lowered the learning rate .Increased the epoch and reduced the batch size . Included Dropout layer as a part of regularization technique . Previous architecture was over fitting on data hence we were higher accuracy on training set and  getting lower accuracy on validation set.

* Which parameters were tuned? How were they adjusted and why?
    Learning rate , epoch , batch_size , No. of convulation layers .Above mentioned parameters were tried and tested on validation with various values and best resulting values were kept in final model to achieve best results .

Why it was adjusted :
---
    Learning rate - as learning rate directly effects calculating change in weights when back-propogating 
    epoch - As no of epoch means no of times we are going through the data in forward-pass and backward pass . So it is important to choose well as lower or highering epoch can make changes in model
    batch_size -As Batch - size defines how much data at a time in memory so we need to modify as per our memory capacity
    keep_prob -  as in drop out we need to mention the amount of neurons we need to skip while calculations . this prameter has impact on training model
    
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

Qualities that might be difficult to classify a image are discussed below:

Image 1 : The white strip part in can be common to other images in training but shape and edges for this image can be detected as shape is unique it has positive point on that.

Image 2 : There are many similar image like this only the central portion can be different in each signs . Background can be varying in each case . So Background and similarity can be the qualities which can cause difficulty in classification. 

Image 3 : The arrow part can be similar to other images but as image sign has its different shape , it can be distinguished and similar shapes may get confused in classification .

Image 4 : As many images have similar edges and shapes of triangular boundary with red as dark part and white as light part that can be a similarity to other images . 

Image 5 : Similar boundary conditions and similar shape can affect the classification of such images .

Image 6 : In this image we have outer circle dark and inner white which is similar to many images with many similar no's as "0" is part of 30,40,20,120 so if in lighting conditions image is unclear it can cause an mis-classification

Above were few points which i considerd major points are :
Boundaries / edges
Similar shapes 
Lightinig conditions 
image distortion 

There can be many more conditions if more variety of images is considered  .

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![Test images with softmax][image8]

![Test images with softmax Chart][image11]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.
The results were in range of above 90's which are very similar to test set which means model is performing well on external images from set of train and validation and test . While there can be similarity in images which can be classified similarly but dominant probability was for correct classified image .

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 2nd last and last cell of the Ipython notebook.

Input Image label  (12, b'Priority road')
----------------------------------------
    1st Guess :Priority road(12)
    99.9%

    2nd Guess :Roundabout mandatory(40)
    0.02%

    3rd Guess :No vehicles(15)
    5.15%

    4th Guess :No vehicles(15)
    6.07%

    5th Guess :No vehicles(15)
    3.35%

----------------------------------------
Input Image label  (18, b'General caution')
----------------------------------------
    1st Guess :General caution(18)
    91.6%

    2nd Guess :Traffic signals(26)
    7.64%

    3rd Guess :Pedestrians(27)
    0.70%

    4th Guess :Pedestrians(27)
    0.00%

    5th Guess :Pedestrians(27)
    0.00%

----------------------------------------
Input Image label  (34, b'Turn left ahead')
----------------------------------------
    1st Guess :Turn left ahead(34)
    99.9%

    2nd Guess :Keep right(38)
    0.00%

    3rd Guess :Children crossing(28)
    0.00%

    4th Guess :Children crossing(28)
    0.00%

    5th Guess :Children crossing(28)
    0.00%

----------------------------------------
Input Image label  (11, b'Right-of-way at the next intersection')
----------------------------------------
    1st Guess :Right-of-way at the next intersection(11)
    83.5%

    2nd Guess :Beware of ice/snow(30)
    16.4%

    3rd Guess :Children crossing(28)
    0.00%

    4th Guess :Children crossing(28)
    0.00%

    5th Guess :Children crossing(28)
    0.00%

----------------------------------------
Input Image label  (38, b'Keep right')
----------------------------------------
    1st Guess :Keep right(38)
    100.%

    2nd Guess :Turn left ahead(34)
    4.26%

    3rd Guess :Speed limit (20km/h)(0)
    3.66%

    4th Guess :Speed limit (20km/h)(0)
    1.74%

    5th Guess :Speed limit (20km/h)(0)
    1.36%

----------------------------------------
Input Image label  (1, b'Speed limit (30km/h)')
----------------------------------------
    1st Guess :Speed limit (30km/h)(1)
    99.8%

    2nd Guess :Speed limit (50km/h)(2)
    0.12%

    3rd Guess :Speed limit (20km/h)(0)
    0.03%

    4th Guess :Speed limit (20km/h)(0)
    0.01%

    5th Guess :Speed limit (20km/h)(0)
    0.00%

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


