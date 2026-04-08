The source code implements the tracking method in 
"Heng Fan, Haibin Ling, Parallel Tracking and Verifying: A Framework for Real-Time and High Accuracy Visual Tracking, ICCV, 2017".
Copyright @ Heng Fan and Haibin Ling, 2017.

Note that the code is serially implemented in Matlab 2016a on Windows 10. For the parallel version, please check out the C++ PTAV.

How to run the code:

1. Install Windows Caffe at https://github.com/happynear/caffe-windows.

2. Install the siamese networks for verifier. The deploy document and caffe model are located in the serial_ptav_v1\siamese_networks\. For details on how to install the caffe, please see how to use SINT at https://staff.fnwi.uva.nl/r.tao/projects/SINT/SINT_proj.html.

3. Just run the script run_PTAV.m

Note: the parameters for fDSST tracker is borrowed from the original paper. Do NOT change!

If you have any questions, please feel free to contact Heng Fan (hengfan@temple.edu)