
# Traffic Sign Recognition

## 0 Project Goal

In this project, we'll build a image classification on German traffic signs. The pictures are given as pickled dataset which had been cleaned to same size and one traffic sign per picture. 

The procedures are:
- Load data, and explore & visualize it;
- Pre-process the data;
- Design the model architecture;
- Evaluate the test result, refine model parameters, until accuracy >93%;
- Test the model on new images, resize the images to (32,32,3) dataset;
- Predict the new images, and calculate the accuracy.

## 1 Load the data

---
[//]: # (Image References)

[image1]: ./results/Labels_SampleSize.png 
[image2]: ./results/orgin_size.png
[image3]: ./results/one_img_one_label.png
[image4]: ./results/one_img_one_label_pp.png

---


### 1.1 Dataset Exploration

The pickled data is a dictionary with 4 key/value pairs:
- `'feature'`: 4D array with 4 dimensions as (num_examples, width, height, channels)
- `'labels'`: 1D array with 1 dimension as (label)
- `'sizes'`: tuples of 2 dimensions as (original width, original height)
- `'coords'`: tuples of 4 dimensions of (x1, y1, x2, y2), representing the coordinations of a bounding box around the sign in the original image

There are 34799 training data, 4410 evaluation data, and 12630 testing data.<br>
Each data has the same size of (32,32,3), 32 in width, 32 in height, and 3 RGB color channels.<br>

There are 43 classes of images, and they were labeled as a number from 0 to 42.

### 1.2 Dataset visualization
To better understand the dataset, 3 graphs were drawn:<br> 

- **quantities vs. label**: several classes of images have less sample size than others, e.g. class 1,6,19,24,27,29,32,37,41,42
![test][image1]
- **original size**: Most of them are close to square, and the size varied from 20x20 to 250x250
![alt text][image2]
- **one image each label**: one image was shown to represent what each label looks like
![alt text][image3]

## 2 Model Architecture

---
[//]: # (Image References)

[image4]: ./results/one_img_one_label_p.png
[image5]: ./results/learning_curve_2.png
[image6]: ./results/learning_curve.png
[image7]: ./results/learning_curve_3.0.png

---

### 2.1 Pre-process

The pickled dataset, as a matter of fact, is quite clean and ready to go. To optimize the computation efficiency, 4 steps would be implemented:
- **Matrix dimensionality reduction (3 colors to 1 grayscale)**: Downgrade 32x32x3 RGB image to 32x32 gray image, will about 2 times faster the computing speed. 
- **Histograms equalization**：the brightness varies from image to image, by histograms equalization, the output would dim down the bright images and dim up the dark images.
- **Image normalization (0~255 to -1~1)**: Normalize the pixels into -1~1, would balance the impact of various contrast from similar images. 

- **One hot encode**: encode the integer labels into unique matrix.

Here shown the pre-processed results:
![alt text][image4]


### 2.2 Model Architecture 

Based on Lenet model, I build the architecture described in below table.
Functions were built to simplify the written.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| stride 1x1, padding same, outputs 32x32x6 	|
| ELU					|												|
| Max pooling 2x2    	| stride 2x2, padding same, outputs 16x16x6 	|
| Convolution 5x5	    | stride 1x1, padding valid, outputs 12x12x16   |
| ELU                  |                                               |
| Max pooling 2x2       | stride 2x2, padding same, outputs 6x6x16      |
| Flatten               | outputs 576                                   |            
| Fully connected       | outputs 250        			                |
| ELU                  |                   			                |
| Dropout               |                   			                |
| Fully connected       | outputs 120        			                |
| ELU                  |                   			                |
| Dropout               |                   			                |
| Fully connected       | outputs 43        			                |
| Softmax				|           									|

### 2.3 Train the Model

To train the model, I used
- parameters: 
        - 10-15 Epochs
        - batch size of 128
        - keep_probablity of 0.45
        - learning rate 0.001
- All layers' initialized weights were set to a random number close to 0, and all layers' initialized bias were set to 0.

### 2.4 Evaluate the Model

At begining, I copied Lenet model, with 3 covnets and 1 flattern layer, and then followed 3 fully connected layer. expect the last one, each of them was activated by 'elu'.


The result shown overfitting, where the training accurary went up to 0.995 and the validation accuracy stayed at no more than 0.85. (I hadn't considered the test accuracy at that moment)
![alt text][image5]

Simply adding one dropout after each of fully connected layer, and adjusting the keep probabilities slowly from 0.90 to 0.45, the final accuracy were about 0.995 train, 0.951 validation and 0.934 test. Which looked fairly good.
![alt text][image7]

## 3 Test the Model on New Image

---
[//]: # (Image References)

[image9]: ./results/NWI.png
[image10]: ./results/NWI_pp.png

---

### 3.1 New Images Creation

12 images were chosen from internet, loaded by cv2, and stored as ndarray.

Then, the images were changed color channel from BRG to RGB, resized to 32x32, and were ready to plotted by matplotlib.
![alt text][image9]


The images are chosen because:
- No. 0,7: images are tilted, and partially covered by word mark;
- No. 5, 11: iamges are titled
- No. 4,12,13: images are partially shaded by snow or people;
- No. 1,5,8: images with background clutter
the rest of images are relatively clearer images.


Examined the images rightly paired to the labels, the same pre processing were implemented, and the result shown as:  
![alt text][image10]

### 3.2 Predict to Each Image
Using the saved model, a new session was run for the new images. The prediction accuracy was only 0.600, much lower than the test accuracy. 
- New_Web_Image real labels: [40 13 13 33 18 **40** 11 **27** 25 **25** 32 **12** 18 **23 14**]
- New_Web_Image pred lables: [40 13 13 33 18 **12** 11 **18** 25 **30** 32 **13** 18 **19 8**]


Interestingly, most of the wrongly predicted images seem to be normal:
- No. 5 (correct:40 pred:12) is clearly presented without clutter or blur;
- No. 7 (correct:27 pred:18) has word mark covered in small area, which won't be a disturbance to human eye;
- No. 11(correct:12 pred:12) is clearly presented without clutter or blur;
- No. 14(correct:14 pred:8)  is a little blur, but the high contrast makes it quite distinguishable


On the other hand,
- No. 9 (correct:25 pred:30) might be a challenge to the model due to backgroung clutter
- No. 13(correct:23 pred:19) is partially snow_coverd, might be new to the model since the patten might rare in training set;




```python

```
