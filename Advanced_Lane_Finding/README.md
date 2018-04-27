**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration*.jpg  "calibration images"
[image2]: ./test_images/test*.jpg "test images"
[image3]: ./output_images/undistort/undistortcalibration*.jpg "Undistorted"
[image4]: ./output_images/pipeline/test*.png "Color & gradient threshold""Perspective transform""lanes & Radius of curvature"
[video1]: ./test_videos/project_vedio.mp4 "test Video"
[video2]: ./test_videos_output/project_vedio.mp4 "test Video output"



# 1、Camera Calibration
# 2、Distortion correction
steps:
1.利用cv2.findChessboardCorners()函数找到棋盘的各个角点
2.然后利用cv2.calibrateCamera()函数就可以获得投影矩阵mtx,以及校准矩阵dist
3.最后利用cv2.undistort()就可以得到校准后的图像

注：这部分代码在Advanced_Lane_Finding.ipynb中有对应部分

# 3、Color/gradient threshold
steps:
1.先将RGB图像转化为HLS图像然后提取出S通道对应的图像
2.利用阈值讲车道线部分像素置为１
3.利用cv2.Sobel()函数求图像的梯度
4.通过一定阈值将包括车道线在内的有用的部分像素置为１
5.保留满足上述两者之一的部分，合成新的二进制图像

# 4、Perspective transform
就是将一个图像变换为另一个视角下的样子。
steps:

1、首先定义好待转换的部分src以及转换后的对应的位置dst
2、通过利用cv2.getPerspectiveTransform(src, dst)得到变换矩阵
3、利用cv2.warpPerspective()就可以将一幅图像转换为另一个视角下的图像

# 5.Detect lane lines
steps:
1、计算透视变换后的二值图像的下半部分的值的和
2、找到和最大的两个位置的横坐标，即为车道线的底部横坐标
3、然后通过逐个滑动窗口进行搜索，找到每一个滑动窗中像素不为零的位置坐标
4、最后将所有滑动窗得到的像素不为零的横纵坐标集中起来，并进行2次曲线拟合，就得到了左右两侧的车道线

# 6.Radius of curvature

具体的理论见图./output_images/Radius_of_curvature.jpg图像中的推倒，注意需要将像素转化成米

#vedio
对于视频的处理和图像的处理完全一致，读取视频处理的代码如下：
from moviepy.editor import VideoFileClip
from IPython.display import HTML

output = 'test_videos_output/project_video.mp4'
clip1 = VideoFileClip("test_videos/project_video.mp4")
white_clip = clip1.fl_image(main_fuction) #NOTE: this function expects color images!!
%time white_clip.write_videofile(output, audio=False)

显示处理完的视频代码如下：
HTML("""
<video width="640" height="360" controls>
  <source src="{0}">
</video>
""".format(output))



