# **Traffic Sign Recognition** 



#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
(34799,32,32,3)
* The size of the validation set is ?
(4410,32,32,3)
* The size of test set is ?
(12630,32,32,3)
* The shape of a traffic sign image is ?
(32,32,3)
* The number of unique classes/labels in the data set is ?
(43)
#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
[image_1]
It is a bar chart showing how the classes distribution in traffic-sign dataset.
[image_2]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I make a compare experiment with both rgb_images and grayscale.I find the accuracy is better in grayscale.

As a last step, I use equalizeHist and normalization with the image data because it can make pixels more balanced distribution.
  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5X5    	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	
| drop			| 0.7 keep_prob
| Convolution 5X5       | 2X2 stride, valid padding,outputs 10X10X16
| RELU   		
| Max pooling	      	| 2x2 stride,  outputs 5x5x16							|
| Fully connected       | 400,120 shape
| Fully connected       | 120,84  shape   
| Fully connected       | 84,10   shape    									|
| Softmax			      									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the cross_entropy as loss,Adam as the optimizer,0.001 as the initial learning_rate,20 epochs and 128 batch_size.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 94.3% 


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
Lenet is the first architecture,but it alse is the final architecture.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:[new_images]dir


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop     		        | No passing 									| 
| Speed limit (50km/h)          | Speed limit (50km/h)										|
| Keep right			| Keep right											|
| Go straight or right	      	| Go straight or right					 				|
| No entry			| Speed limit (30km/h)     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were
first image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9        			        | No passing 									| 
| .1     				| Priority road										|
| .0				        | Speed limit (100km/h)											|
| .0	      			        | Turn right ahead					 				|
| .0			                | No vehicles     							|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


