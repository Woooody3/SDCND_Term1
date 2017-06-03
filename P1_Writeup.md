
# P1 **Finding Lane Lines on the Road** - Writeup 

---
## Summary

First step to autonomous driving is to understand what we see. Marking out the lane line are one of the visual elements. Here are steps:
- mark the lane line in one image (pipeline):
    - read in the image as pixel matrix 
    - turn color RGB into gray, to lower degrees
    - smooth the image with Gaussian method 
    - find out the edges via Canny Edge Detection: white in edges, black in others
    - define the region of interest, fill the outsider area with black 
    - connect the white edge dots into lines via Hough method
    - refine the lines into one left line and one right line, and draw on the original picture
- repeat marking on the whole video.

---
## Reflection

### 1. Pipeline Version 1

**Implementation** <br>
With `Helper Functions`, the first pipeline is able to identify the lane edges in the area we concerned. Followed parameters setting and process steps:


```python
def pipeline(image):
    # set values for parameters
    kernel_size = 5
    low_threshold = 50
    high_threshold = 150
    
    imshape = image.shape
    #vertices = np.array([[(0,imshape[0]),(0,0),(imshape[1],0),(imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(0,imshape[0]),(imshape[1]/2,imshape[0]*.59),(imshape[1],imshape[0])]], dtype=np.int32)
    
    rho = 2
    theta = np.pi/180
    threshold = 50
    min_line_len = 20
    max_line_gap = 50
 
```


   * RGB to GRAY  ![RGB to GRAY][image1]
   * Gaussion smoothing ![Gaussion][image2]
   * Canny edge detection ![Edge][image3]
   * Highlight ROI and mask image ![Mask][image4]
   * Hough Line ![Line][image5]

In order to refine the small lines into 2, I modified the `draw_line()` functon:
- initialize the left line (x1, y1, x2, y2) and right line(x1, y1, x2, y2)
- find out the value
- draw the line

[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve_gray.jpg "1"
[image2]: ./test_images_output/solidWhiteCurve_gray_blur.jpg "2"
[image3]: ./test_images_output/solidWhiteCurve_edge.jpg "3"
[image4]: ./test_images_output/solidWhiteCurve_edge_mask.jpg "4"
[image5]: ./test_images_output/solidWhiteCurve_line.jpg "5"
[image6]: ./test_images_output/solidWhiteCurve_line2.jpg "6"




```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=15):
    ## initialization
    yr_max = 0
    yr_min = img.shape[0]
    xr_max = 0
    xr_min = img.shape[1]
    
    yl_max = 0
    yl_min = img.shape[0]
    xl_max = 0
    xl_min = img.shape[1]
    
    ## segment left and right line
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1))>0:           #right segment
                yr_min = min(y1, y2, yr_min)
                yr_max = max(y1, y2, yr_max)
                xr_min = min(x1, x2, xr_min)
                xr_max = max(x1, x2, xr_max)
                
            else:                             #left segment
                yl_min = min(y1, y2, yl_min)
                yl_max = max(y1, y2, yl_max)
                xl_min = min(x1, x2, xl_min)
                xl_max = max(x1, x2, xl_max)
                
    cv2.line(img, (xr_min, yr_min), (xr_max, yr_max), color, thickness) #right line
    cv2.line(img, (xl_max, yl_min), (xl_min, yl_max), color, thickness) #left line
```

**Problems** <br>
It is tested fail on video `solidYellowLeft.mp4` that:
- errors lines in some image; ![Error1][image7]
- lines in dotted portion varies in length ![Error2][image8]



[//]: # (Image References)
[image7]: ./test_images_output/error_1.png "1"
[image8]: ./test_images_output/error_2.png "8"

---
### 2. Pipeline Version 2

**Problem solving_1** <br>

Error 1 occured because lack of detection to small dotted lane marks, comparing to `raw-lines-example.mp4`. To solve that, the parameters need to be fine-toned:<br>   
    1\. theta decreased half: to have smaller grid in Hough transformation and thus the higer accuracy       
    2\. lower the threshold from 50 to 20  
    3\. lower the min_line_len from 20 to 5  
    4\. lower the max_line_gap from 50 to 20  

item 2~4 are aim to increase the ability of recognizing shorter lines



```python
def pipeline(image):
    # set values for parameters
    kernel_size = 5
    low_threshold = 50
    high_threshold = 150
    
    imshape = image.shape
    #vertices = np.array([[(0,imshape[0]),(0,0),(imshape[1],0),(imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(0,imshape[0]),(imshape[1]/2,imshape[0]*.59),(imshape[1],imshape[0])]], dtype=np.int32)
    
    rho = 2
    theta = np.pi/360
    threshold = 20
    min_line_len = 5
    max_line_gap = 20
    
```

**Problem solving_2** <br>

To solve error2, `draw_line()` functon should be updated. Instead of finding out the max/min of lines as the end points, here I:
- fixed the end point(x,y) with y value range same as the y range in region of interest
- seperate lines into two groups by slope, (0, 2) marks as right segment, and (-2, 0) left segment;
- in each segment, average the lines to get the output slope, and center point
- calculate the endpoint's (x1, y1, x2, y2), and draw line



```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    ## initialization
    i = y_sum = x_sum = s_r = 0
    j = yl_sum = xl_sum = s_l = 0
    y_min = img.shape[0]*.6
    y_max = img.shape[0]
    
    ## segment left and right line
    for line in lines:
        for x1,y1,x2,y2 in line:                
                if 2>((x2-x1)/(y2-y1))>0: #right segment
                    i += 1
                    y_sum += y2 + y1
                    x_sum += x2 + x1
                    s_r += (x2-x1) / (y2-y1)
                
                elif 0>((x2-x1)/(y2-y1))>-2: #left segment
                    j += 1
                    yl_sum += y2 + y1
                    xl_sum += x2 + x1
                    s_l += (x2-x1) / (y2-y1)
    if j == 0:
        print("there is no left line")
    if i == 0:
        print("there is no right line")
    
    ## calculate right line start and end point
    s_r = s_r / i  
    b_r = (x_sum - s_r * y_sum)/ (2*i) 
    xr_min = int(s_r * y_min + b_r)
    xr_max = int(s_r * y_max + b_r)  
    cv2.line(img, (xr_min, int(y_min)), (xr_max, y_max), color, thickness) #right line
    
    ## calculate left line start and end point
    s_l = s_l / j
    b_l = (xl_sum - s_l * yl_sum)/ (2*j)
    xl_min = int(s_l * y_min + b_l)
    xl_max = int(s_l * y_max + b_l)  
    cv2.line(img, (xl_min, int(y_min)), (xl_max, y_max), color, thickness)#right line
```

 And the output looks better, passed the 2 video tests, `solidWhiteRight.mp4` and `solidYellowLeft.mp4`.
 
 **`./test_videos_output/solidWhiteRight.mp4` is the result output from Pipeline V2.**
 
- For most of time, it detects lines fine; ![PipelineV2_solidYellowLeft_VideoClip][image1]
- however, lines are trembling in the video, and sometimes even offtracking from the lane mark   ![PipelineV2_solidYellowLeft_VideoClip_2][image2]


[//]: # (Image References)
[image1]: ./test_images_output/PipelineV2_solidYellowLeft_VideoClip.png "1"
[image2]: ./test_images_output/PipelineV2_solidYellowLeft_VideoClip_2.png "8"

---
### 3. Pipeline Version 3

**Problem solving** <br>

To stablize the lines and diminish the offtracking, `draw_line()` functon should be further fine-toned. Together with "slope filter", one "intersept filter" is added to select the lines which corretly represented the lane <br> 
In every line having a close slope to right lane, I do:
* calculate the line's intersept to x, b_i
* compared b_i with cumulated lines' intersept to x, b_r/i
    * if the line has a similar intersept, accumulate it to b_r
    * if the line has a far-away intersept, it's not close to right lane, should be eliminated
* filters narrow to slope range(-2,2), intercept range(-20,20)
Same procedure applied to lefe segment. <br>
With both slope and intersept filters, the noise lines could be excluded to final line output.


```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    ## initialization
    i = yr_mid = xr_mid = s_r = b_r = 0
    j = yl_mid = xl_mid = s_l = 0
    y_min = img.shape[0]*.6
    y_max = img.shape[0]
    b_l = img.shape[1]
     
    ## segment left and right line
    for line in lines:
        for x1,y1,x2,y2 in line:             
            if 2>((x2-x1)/(y2-y1))>0:        ####right segment
                i += 1
                yr_mid += (y2 + y1)/2        ####accumulate the center of y              
                xr_mid += (x2 + x1)/2        ####accumulate the center of x            
                s_r += (x2-x1) / (y2-y1)     ####accumulate the value of slope
                b_i = ((x2+x1) - (y2+y1) * (x2-x1)/(y2-y1)) / 2  ####intercept b for current line i
                if abs(b_i - b_r/i) < 20:    ####accumulate the value of intercept
                    b_r = xr_mid - s_r/i * yr_mid 
                else:                        ####intercept b differs hugely should be eliminated
                    i -= 1
                    yr_mid -= (y2 + y1)/2
                    xr_mid -= (x2 + x1)/2
                    s_r -= (x2-x1) / (y2-y1)
                
            elif 0>((x2-x1)/(y2-y1))>-2:     ####left segment
                j += 1
                yl_mid += (y2 + y1)/2
                xl_mid += (x2 + x1)/2
                s_l += (x2-x1) / (y2-y1)
                b_j = ((x2+x1) - (y2+y1) * (x2-x1)/(y2-y1)) / 2
                if abs(b_j - b_l/j) < 20:
                    b_l = xl_mid - s_l/j * yl_mid
                else:
                    j -= 1
                    yl_mid -= (y2 + y1)/2
                    xl_mid -= (x2 + x1)/2
                    s_l -= (x2-x1) / (y2-y1)

    if j == 0:
        print("there is no left line")
    if i == 0:
        print("there is no right line")
    
    ## calculate right line start and end point
    xr_min = int((s_r * y_min + b_r)/i)
    xr_max = int((s_r * y_max + b_r)/i)  
    cv2.line(img, (xr_min, int(y_min)), (xr_max, y_max), color, thickness) #right line
    
    ## calculate left line start and end point
    xl_min = int((s_l * y_min + b_l)/j)
    xl_max = int((s_l * y_max + b_l)/j) 
    cv2.line(img, (xl_min, int(y_min)), (xl_max, y_max), color, thickness)#right line

```

**`./test_videos_output/solidYellowLeft.mp4` is the result output from Pipeline V3.** The stabalization improves from pipeline v2 solidWhiteRight.mp4.


---
### 4. Pipeline Version 4 (Test on Chanllenge.mp4)
**Problem solving** <br>

Apply Pipeline V3 to `Chanllenge.mp4` will throw errors. It failed to detect left or right lines because the initial intercept value were put by my "best guess". Here the camera view angle changes, follows the lane intercept value range. I update the calculating concepts as:
* calculate the line's intersept to x, b_i
* find out if current line as the first right line (i == 1)
    * if so, begin to accumulate intersetp as b_r = b_i
    * otherwise, move to next step
    
    * compared b_i with cumulated lines' intersept to x, b_r/(i-1)
    * if the line has a similar intersept, accumulate it to b_r
    * if the line has a far-away intersept, it's not close to right lane, should be eliminated 

Same procedure applied to lefe segment. <br>



```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    ## initialization
    i = yr_mid = xr_mid = s_r = b_r = 0
    j = yl_mid = xl_mid = s_l = b_l = 0
    y_min = img.shape[0]*.6
    y_max = img.shape[0]
   
    ## segment left and right line
    for line in lines:
        for x1,y1,x2,y2 in line:             
            if 2>((x2-x1)/(y2-y1))>0:        ####right segment
                i += 1
                yr_mid += (y2 + y1)/2        ####accumulate the center of y              
                xr_mid += (x2 + x1)/2        ####accumulate the center of x            
                s_r += (x2-x1) / (y2-y1)     ####accumulate the value of slope
                b_i = ((x2+x1) - (y2+y1) * (x2-x1)/(y2-y1)) / 2  ####intercept b for current line i
                if i == 1:                   ####initialize the b_r
                    b_r = b_i
                else:
                    if abs(b_i - b_r/(i-1)) < 20:    ####accumulate the value of intercept
                        b_r = xr_mid - s_r/i * yr_mid
                    else:                        ####intercept b differs hugely should be eliminated
                        i -= 1
                        yr_mid -= (y2 + y1)/2
                        xr_mid -= (x2 + x1)/2
                        s_r -= (x2-x1) / (y2-y1)
                
            elif 0>((x2-x1)/(y2-y1))>-2:     ####left segment
                j += 1
                yl_mid += (y2 + y1)/2
                xl_mid += (x2 + x1)/2
                s_l += (x2-x1) / (y2-y1)
                b_j = ((x2+x1) - (y2+y1) * (x2-x1)/(y2-y1)) / 2
                if j == 1:
                    b_l = b_j
                else:                    
                    if abs(b_j - b_l/(j-1)) < 20:
                        b_l = xl_mid - s_l/j * yl_mid
                    else:                        
                        j -= 1
                        yl_mid -= (y2 + y1)/2
                        xl_mid -= (x2 + x1)/2
                        s_l -= (x2-x1) / (y2-y1)

    if j == 0:
        print("there is no left line")
    if i == 0:
        print("there is no right line")
    
    ## calculate right line start and end point
    xr_min = int((s_r * y_min + b_r)/i)
    xr_max = int((s_r * y_max + b_r)/i)  
    cv2.line(img, (xr_min, int(y_min)), (xr_max, y_max), color, thickness) #right line
    
    ## calculate left line start and end point
    xl_min = int((s_l * y_min + b_l)/j)
    xl_max = int((s_l * y_max + b_l)/j) 
    cv2.line(img, (xl_min, int(y_min)), (xl_max, y_max), color, thickness)#right line

    
```

With both slope and intersept filters, the noise lines could be excluded to final line output.
**`./test_videos_output/challenge_PipelineV4.mp4` is the result output from Pipeline V4.**

---
### 5. Future Possible Improvements Idea

Several improvements are open to be enhanced still:
* line stabalization: 
    * the result shows line shift from lane in some cases, and is generally more trembling than the sample video. 
    * the shift might be caused by intersept initialization with first line in the matrix is offtrack 
    * a possible improvement might be to define the slope and intercept initial value as average for all previous lines.
    
* computing efficiency:
    * the function `draw_line()` works, but it seems to not be in high efficiency
    * a possible improvement might be to describe the line with 2 parameters (theta and rho) instead of 4 parameters (x1,x2,y1,y2). simpler fiters and simper computing.



```python

```
