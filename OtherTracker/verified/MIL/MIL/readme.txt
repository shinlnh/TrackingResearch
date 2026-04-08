----------------------------------------------------------------------------------------------------------------------------------------------
* Readme of the MATLAB demo for paper "Visual Tracking with Online Multiple Instance Learning" by       Boris Babenko, Ming-Hsuan Yang, Serge Belongie ,CVPR 2009, Miami, Florida.
* Author: Kaihua Zhang, The Hong Kong Polytechnic University
* Email: zhkhua@gmail.com
* version 1.0
* Date: 26/8/2012.
-----------------------------------------------------------------------------------------------------------------------------------------------
* Note: the purpose of this code is only for better understanding of MILTrack. The results by this code 
   may be different from Boris' C++ code. Use it at your own risk.
-----------------------------------------------------------------------------------------------------------------------------------------------
Procedures
>Put the video sequences into file ¡®\data¡¯; 
>Initialize the position in the first frame in ¡®\data\init.txt¡¯; The setup has the format  [x y width height] where ¡°x,y ¡± are the coordinate of left top point of the rectangle.
> run ¡°mexCompile.m¡± to generate ¡°mex¡± files 
> run ¡°runTracker.m¡±
-----------------------------------------------------------------------------------------------------------------------------------------------
Tracking results will be saved to file "MILTrackResults.txt".