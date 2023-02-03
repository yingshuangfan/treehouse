#!/usr/bin/env python
# coding: utf-8

# # Computer Vision
# 
# ## ResNet - Deep Residual Learning for Image Recognition
# 
# Intuition: Deep network is hard to optimize. E.g. AlexNet, VNN. Why does the increase of layers lead to higher training errors? 
# 
# Suppose a shallow network learned a function f, then we could assume that a deeper network is capable of learning the same function by copying the first N-layers exactly as the shallow network and the rest of the layers which simulate an identity matrix. In other words, the accuracy of the deeper network should reach (or be close) to the shallow network. Why can't these extra layers learn the identity matrix? 
# 
# Notice: The problem we met here is different from overfitting! The bad performance is NOT due to an increase in parameters. On the other hand, the solution space of the shallow network is the subset of the deep network. Therefore, we should be capable of learning the exact same function on a deeper network or even gain a better performance(on the training set).
# 
# ### Explanation:
# 
# Initialization: values are randomly initialized around zero. Thus learning an identity function is NOT any easier to learn any other function.
# 
# **Hypothesis:** Use the identity function as the default value instead of zeros.
# 
# **How:** Add a "skip" connection from the input to the output(with ReLU), thus the default function would be the identity function since the output is identical to the input if the weight layers do not learn anything. This is the so-called residual connection. However, if the shape of the output is different from the input, there are different options to design a valid connection:
# 
# A. Zero-padded.
# 
# B. One-by-one convolution.
# 
# C. One-by-one convolution + ???(introduce much more parameters)
# 
# ### Remark:
# 
# - The performance of option 3 only enhances the accuracy marginally, thus option 2 is the default strategy now in the industry.
# - The deeper the network, the more helpful the residual connections are. 
# - Ensemble the ResNets won the ImageNet contest that year.
# 
# Reference: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

# ## You Only Look Once: Unified, Real-Time Object Detection
# 
# **Intuition:** Split the image into N*N grid cells, with each cell be presented with a fixed length of vector. The vector contains information as follows:
# 
# - Pc: if the cell contains an object; if Pc=0, then the rest of the values don't matter.
# - Bx, By: the center coordinate of the  bounding box; the center should always lies in the cell, thus Bx and By are from 0 to 1.
# - Bw, Bh: the width and height of the bounding box; the box could extend  out of the cell, thus Bw and Bh could be larger than 1.
# - Ci: the category id; this could be one-hot encoding.
# 
# ### Remark:
# 
#  - YOLO algorithm is very fast compared to algorithms such as CRNN. The main reason lies in the name: because we only need to traverse each image for one time. 
#  - The method is capable of representing multiple objects as long as the number of cells is sufficient. 
# 
# ### Extended Discussion: 
# 
# **Q1.** When predicting a new image, the algorithm could output multiple potential bounding box for the same object. How to select the best one?
# 
# **Solution:** Non-max suppression, or IOU. The method compares each pair of bounding box if they are overlapping, and discard one of them if the overlapping area is greater than the threshold. After IOU, the each cell should output a unique bounding box if applicable.
# 
# **Q2.** What if there are multiple objects in a single grid cell?
# 
# **Solution:** Anchor box, with two bounding boxes and category ids in a single vector. The vector would be fixed length and thus has the capability of representing at most two objects. The solution is effective because generally the grid cells are so small that they cannot contain multiple objects. Therefore, as a result an 2-anchor box would be sufficient enough in real-world scenarios.
# 
# Reference: Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
# 
