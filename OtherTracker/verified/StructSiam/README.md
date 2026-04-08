

# Introduction

This is the test code of "Structured Siamese Network for Real-Time Visual Tracking" for your dataset 

# Prerequisites

python 2.7

ubuntu 14.04

cuda-8.0

cudnn-6.0.21

[Tensorflow-1.3-gpu](https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/gpu/tensorflow_gpu-1.3.0rc0-cp27-none-linux_x86_64.whl)

NVIDIA TITAN X GPU

 

# Notice

* Run the code "python tracking_test.py". 
* You need to modify 'sequences_dir' in the code 'tracking_test.py' to your directory. 
* If your groundtruth files are different from that of OTB, you may need to modify the "_init_video" function in 'tracking_test.py'. 
* The tracking results are recorded in variables "bBoxes" and "speed_i". 