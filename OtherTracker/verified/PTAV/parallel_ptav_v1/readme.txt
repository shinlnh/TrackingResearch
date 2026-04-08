The source code implements the tracking method in 
"Heng Fan, Haibin Ling, Parallel Tracking and Verifying: A Framework for Real-Time and High Accuracy Visual Tracking, ICCV, 2017".
Copyright @ Heng Fan and Haibin Ling, 2017.

Note that the code is implemented in parallel threads in Visual Studio 2013 on Windows 10. For the serial version, please check out the Matlab PTAV.

How to run the code:

1. Install Windows Caffe at https://github.com/happynear/caffe-windows.

2. After successfully compiling the Windows Caffe, create a new project PTAV (you'd better copy the classification project, rename it, and revise the parameters).

3. Add the files in PTAV\ to the project. Note that the runtracker.cpp is the main file in this project, and other files (including dirent.h, ffttools.hpp, fhog.cpp, fhog.hpp, kcftracker.cpp, kcftracker.hpp, labdata.hpp, recttools.hpp, tracker.h, verificator.h) should be header files.

4. Afte step 3, you should be able to compile the project PTAV. In this step, you will need three files, deploy.prototxt, similarity.caffemodel, and image_mean.binaryproto, to load the siamese networks (i.e., the verifier). The image_mean.binaryproto is located in the parallel_ptav_v1\, and the other two files are placed in the serial_ptav_v1.zip.

5. Once you successfully step 4, you can obtain the exe file in the caffe\bin\, named as PTAV.exe

6. Use it in cmd as: PTAV.exe video_name, e.g., PTAV.exe Jogging-1

Note: the origial fDDST tacker is implemented in Matlab. Here we revise to get the c++ interface by compiling it into .dll and .lib files. Use the following comment in Matlab (2016a) to compile:

>> mcc -W cpplib:fDSST -T link:lib fDSST.m

After compiling, you are expected to get some files, and all you need are fDSST.lib, fDSST.dll and fDSST.h. Then, you can call the interface to use it in C++.

If you have any questions, please feel free to contact Heng Fan (hengfan@temple.edu)