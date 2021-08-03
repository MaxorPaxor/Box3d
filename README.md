# RealSense

## Overview

An experimental tool whos purpose is to calculate any cuboid-like object
dimensions. This repo focuses on the use of RealSense Stereo cameras:
D435i, D455 and Lidar L515.

<p align="center">
<img src="images/1.png" >
</p>
<h6 align="center">Object RGB and depth map</h6>

<p align="center">
<img src="images/2.png" >
</p>
<h6 align="center">Painting the object with the mouse</h6>

<p align="center">
<img src="images/3.png" >
</p>
<h6 align="center">Calculating the box dimensions</h6>

### Performance
#### Measurements
Calculated Dimensions [mm] | Real Dimensions [mm] | Error (%)
:---:|:---:|:---:
89.8 | 95 | 5.5
408.7 | 415 | 1.5
282.9 | 290 | 2.4

#### Repetitiveness
The algorithm right now is not robust enough to handle changes in
distances, human selecting variance (different operators) etc

## References
Based on RealSense SDK:
https://github.com/IntelRealSense/librealsense

With python wrapper:
https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python

pip3 install pyrealsense2

pyRANSAC is used for 3 orthogonal planes RANSAC
https://github.com/leomariga/pyRANSAC-3D

pip3 install pyransac3d

Open3D lib for debugging and visualing:
http://www.open3d.org/

pip3 install open3d

## Algo Review