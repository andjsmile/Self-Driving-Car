# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 
- grayscale

- canny（）

- gaussian_blur（）

- region_of_interest（）

- draw_lines（）

- hough_lines（）

I modified the draw_lines() function by calculating slope for judging which side is right or left,then I get the center points and slopes of both sides ，then I calculating the bottom points and the top points by the center points and slopes .Finally,I draw lines.




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be the video of" Improve the draw_lines()" can't load correctly.

Another shortcoming could be my code can't have a good with the third part--Optional Challenge.


### 3. Suggest possible improvements to your pipeline

I have tried my best to this project,but the second vedio can not work correctly ,I hope you can give me some good advice to modify it ,and I hope you can give me some Practical advice for the third vedio,thanks.
