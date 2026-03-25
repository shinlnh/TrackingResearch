
1. Download the VGG-NET-19 mat file using the link

https://uofi.box.com/shared/static/kxzjhbagd6ih1rf7mjyoxn2hy70hltpl.mat
or using the link if you are in China
http://pan.baidu.com/s/1kU1Me5T 

and put it into some folder.


Note that this mat file is compatile with the MatConvNet-1beta8 used in this work, if you download the mat file from
http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat,
please pay attention to the version compatibility. You may need to modify some names of fields in each convolutional layer.  

2. Using the preCompiled Matconvnet (not recommended) or Compile yourself Matconvnet using Matlab
in the command window, type 

>>cd matconvnet1.08
>>addpath matlab
>>vl_compilenn('enableGpu', true)

Waiting the notification of success.
More information about Matconvnet can be found at http://www.vlfeat.org/matconvnet/install/

3. Comment the first line and config paths in 'run_HDT.m' according to your environment.

4. Run 'run_HDT.m'. 

*********************************************************************************
If you find the code helpful in your research, please consider citing:

@inproceedings{HDT-CVPR-2016,
    title={Hedged Deep Tracking},
    Author = {Yuankai Qi and 
              Shengping Zhang and
              Lei Qin and
              Hongxun Yao and
              Qingming Huang and
              Jongwoo Lim and
              Ming-Hsuan Yang},
    booktitle = {CVPR},
    pages = {},
    Year = {2016}
}
*********************************************************************************

Feedbacks and comments are welcome! 
Feel free to contact us via qykshr@gmail.com or s.zhang@hit.edu.cn.


