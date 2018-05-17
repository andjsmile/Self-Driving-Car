**Vehicle Detection Project**


总体描述：这个项目是做了一个利用Hog+SVM进行视频中车辆的跟踪，全程没有使用任何神经网络的算法．最后的效果展示见project_result.video.

对Hog特征几点要说明的：
１．其主要思想是将图像分块，进而计算每个块中的梯度．用梯度组合来表示此类物体的特征
２．参数说明：sklearn中包含了hog函数，from skimage.feature import hog；
　　其中重要的参数有以下几个：orientations, pixels_per_cell and cells_per_block,transform_sqrt
　　　 orientations:orientations的数量被指定为一个整数，并且表示梯度信息将被分割到直方图中的方位箱的数量。
    典型值在6到12个分箱之间;
    pixels_per_cell:指定计算每个梯度直方图的单元大小。这个参数是作为一个2元组传递的，所以你可以在x和y中有不同的单元大小，
    但是通常选择单元格为正方形。
    cells_per_block:cells_per_block参数也作为2元组传递，并指定局部区域，其中给定单元格中的直方图计数将被归一化。
    transform_sqrt:这种标准化可能有助于减少阴影或其他光照变化的影响，但如果图像包含负值（因为它取图像值的平方根），将导致错误。
此部分代码如下：
######################################################################################
features = hog(　img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
######################################################################################
其中我使用的参数如下：
			orient = 15  # HOG orientations
			pix_per_cell = 8 # HOG pixels per cell
			cell_per_block = 2 # HOG cells per block
			
此外，在此项目中，除Hog，我还利用了Spatial Binning of Color和Histograms of Color．这两种方法利用的是图像的颜色信息，能够结合Hog利用空间信息更好的帮助SVM算法进行分类．具体代码如下：
######################################################################################
Histograms of Color:
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    #print(bin_edges)
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    #print(bin_centers)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features
######################################################################################
Spatial Binning of Color:
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2LUV)
    #img=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    # Use cv2.resize().ravel() to create the feature vector
    img=cv2.resize(img, size)
    features = img.ravel() # Remove this line!
    # Return the feature vector
    return features
######################################################################################
对滑动窗口方法的讨论：
为了每次仅对一幅图像进行一次Hog特征提取，这里用到了＂子采样窗口搜索＂的方法．即每次对一幅图像先进行整体的Hog特征计算，然后选取适当大小的滑动窗逐次在图像上移动；每次仅选取滑动窗内部的Hog特征进行图像判别．为了更合适的框住车辆，我们采用不同尺度的scale率先对图像的长宽做伸长或者压缩，为了能用不同大小的框尽可能更适合的框住车辆．
在滑动窗方法进行完，可能会出现误判别的情况，即没有车辆的地方被框住；也会出现在同一车的周围出现多个框．为了消除这两个问题，这里采用了heatmap的方法．即每次被框住的像素值加１，最后设定一个阈值，小于等于该阈值，则此框无效．为了更降低误匹配的情况．这里用到了一种策略．即：通过前８帧对前８帧的热图进行记录，如果到达８帧以后，连续三帧一个位置被检测到车辆，则才认为此位置的确有车辆，利用这种策略进一步降低了误识别率．代码如下：
######################################################################################
　　　　history = deque(maxlen = 8)
　　　　heat = np.zeros_like(out_img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, bboxes)

    # Apply threshold to help remove false positives
    threshold = 1 
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    current_heatmap = np.clip(heat, 0, 255)
    history.append(current_heatmap)
    heatmap = np.zeros_like(current_heatmap).astype(np.float)
    for heat in history:
        heatmap = heatmap + heat
    if len(history)==8:
        threshold = 3 
        heatmap = apply_threshold(heatmap , threshold)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
######################################################################################









