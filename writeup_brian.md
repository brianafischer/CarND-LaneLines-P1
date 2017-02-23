#**Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of the following main steps:
	1. Convert image to grayscale
	2. Apply noise removal/smoothing via a Gaussian blur
	3. Run a Canny edge detector on the smooth grayscale image
	4. Apply a region of interest that only includes feasible lane markings
	5. Apply a Hough transform to this processed image to get a list of lane markings
	6. Apply a custom algorithm (draw_lines) on the detected lane markings to draw a single left and right lane
	
The main effort was in tweaking the algorithm parameters (represented as constants in the application) along with the draw_lines() function.

The draw lines function takes the average slope of each lane marking to determine a left or right marking.  Then the average of all left/right markings are computed and a single line is drawn using the average values.


###2. Identify potential shortcomings with your current pipeline

There is currently no filtering on the slope to determine if the lane marking is reasonable.  This will allow outliers (lines that could not possibly be lanes) to skew the averaged lane.
In addition, the use of grayscale may be simple but it is most likely not as robust as the HSV color space.  Using the HSV color space to detect strong white and yellow lines in different conditions (shadows, asphalt, …) would most likely have better performance.


###3. Suggest possible improvements to your pipeline

There are many possible improvements, finding the time to implement them is the challenge!

Averaging - Using multiple frames/a history of data could be taken into account to improve the algorithm accuracy.

Linear Lanes - The current algorithm is very basic and only uses linear lines to highlight the lane.  It would be appropriate to use a 3 or 4 point polyline to highlight curved lanes. 

Testing and Robustness - I would like to spend some more time collecting sample videos in different environments.  When no lanes are detected, it would be interesting to predict lanes how humans do (looking for tire tracks and free space).