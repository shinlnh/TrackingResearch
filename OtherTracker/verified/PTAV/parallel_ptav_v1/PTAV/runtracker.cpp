#include <caffe/caffe.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <time.h>
#include <iosfwd>
#include <memory>
#include <string>
#include <windows.h>
#include <utility>
#include <vector>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "kcftracker.hpp"
#include "dirent.h"
#include "verificator.h"

// #include "matlabKCF.h"
#include "fDSST.h"


using namespace caffe;
// using namespace std;
using namespace cv;
using std::string;

// a set of global variables

#ifndef my_max
#define my_max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef my_min
#define my_min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

string  video;               // video name
string  base_path;           // base path containing videos
string  video_path;
int     img_width;
int     img_height;
int     img_sz = 512;
Mat     frame[3500];        // store image frames, change the size of array if necessary
Mat     net_frame[3500];
Rect    result;

int start_num;
int end_num;
int total_frames;
int tracking_point = 1;
int frame_num;

float init_xMin, init_yMin, init_width, init_height;

vector<float> tracking_results[3500];

bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = false;
bool LAB = true;

int V = 10;
int v_idx = 0;
int V_bk = 10;
int init_flag = 0;
int new_frame = -1;
bool new_start = false;
bool trackEnd = false;
bool verifyEnd = false;
bool trackChange = true;
float verify_th = 1.0;
float det_threshold = 1.6;

float lamda = 1.5;
vector<float> new_roi;

Point pt(10, 30);
Scalar color = CV_RGB(0, 0, 255);

// Create KCFTracker object
// Note that this KCFTracker object is just used to display tracking results, and 
// we use the fDSST for tracking part
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB); 

// caffe model, change the path if necessary
string model_file   = "C:\\Users\\HengLan\\Desktop\\caffe-windows-master\\python\\deploy.prototxt";
string trained_file = "C:\\Users\\HengLan\\Desktop\\caffe-windows-master\\python\\similarity.caffemodel";
string mean_file    = "C:\\Users\\HengLan\\Desktop\\caffe-windows-master\\python\\imagenet_mean.binaryproto";

// feature for the object in the first frame, should be a 1 x 54272 matrix
Mat first_feat;  

// get image name
string get_image_name(string path, int i)
{
	stringstream ss;

	if (i < 10)
	{
		ss << 0 << 0 << 0 << i;
	}
	else if (i < 100)
	{
		ss << 0 << 0 << i;
	}
	else if (i < 1000)
	{
		ss << 0 << i;
	}
	else
	{
		ss << i;
	}
	return path + ss.str() + ".jpg";
}


// inisialize some variables
Classifier init_global_vars()
{
	// "E:\\TrackingBenchmark13\\" is the path to your videos
	string tmp_img_file = "E:\\TrackingBenchmark13\\" + video + "\\img\\0001.jpg";
	Mat img = imread(tmp_img_file, CV_LOAD_IMAGE_COLOR);

	img_width = img.cols;
	img_height = img.rows;

	// path to your videos
	base_path = "E:\\TrackingBenchmark13\\";
	video_path = video + "\\";

	// read start frame and end frame
	std::ifstream frame_num_file;
	string frame_num_name = base_path + video_path + "frames.txt";   // "frames.txt" provides the first and last frame number of each video
	frame_num_file.open(frame_num_name);
	string frame_line;
	getline(frame_num_file, frame_line);
	frame_num_file.close();

	std::istringstream ff(frame_line);

	// read start number and end number
	char ch;
	ff >> start_num; ff >> ch; ff >> end_num;

	total_frames = end_num - start_num + 1;   // number of frames

	// Read groundtruth for the 1st frame
	std::ifstream groundtruthFile;
	string groundtruth = base_path + video_path + "groundtruth.txt";
	groundtruthFile.open(groundtruth);
	string firstLine;
	getline(groundtruthFile, firstLine);
	groundtruthFile.close();

	std::istringstream ss(firstLine);

	// Read groundtruth like a dumb
	ss >> init_xMin; ss >> ch; ss >> init_yMin; ss >> ch; ss >> init_width; ss >> ch; ss >> init_height;

	// initialize KCFTracker for displaying
	string frameName;
	frameName = get_image_name(base_path + video_path + "img\\", start_num);
	frame[start_num] = imread(frameName, CV_LOAD_IMAGE_COLOR);

	tracker.init(Rect(init_xMin, init_yMin, init_width, init_height), frame[start_num]);
	rectangle(frame[start_num], Point(init_xMin, init_yMin), Point(init_xMin + init_width, init_yMin + init_height), Scalar(0, 0, 255), 3, 8);

	// load the siamese networks for verifier
	Classifier classifier(model_file, trained_file, mean_file);

	// get the feature for object in the first frame
	vector<float> first_box;
	first_box.push_back(init_xMin); first_box.push_back(init_yMin); first_box.push_back(init_width); first_box.push_back(init_height);
	net_frame[start_num] = imread(frameName, -1);

	vector<float> tmp_pred = classifier.Classify(net_frame[start_num], img_width, img_height, img_sz, first_box);

	Mat tmp_out(tmp_pred);
	Mat tmp_out_reshape = tmp_out.reshape(0, first_box.size() / 4);
	Mat tmp_first_feat = tmp_out_reshape.t();
	first_feat = tmp_first_feat.t();    // first_feat should be a 1 x 54272 matrix

	system("cls");
	return classifier;
}

// get feature for a box in the frame, should be a 54272 x N matrix, and N is the number of boxes
Mat get_network_feature(Classifier c, Mat curr_img, vector<float> box)
{
	vector<float> tmp_pred = c.Classify(curr_img, img_width, img_height, img_sz, box);

	Mat tmp_out(tmp_pred);
	Mat tmp_out_reshape = tmp_out.reshape(0, box.size() / 4);
	Mat tmp_feat = tmp_out_reshape.t();
	return tmp_feat;
}

// get candidates from the surrounding area...will retuen a vector which contains the bounding boxes of candidates
vector<float> get_candidates(vector<float> res_rect, vector<float> last_rect, Mat im_color, float lamda)
{
	vector<float> img_center;
	img_center.push_back(res_rect[0]);
	img_center.push_back(res_rect[1]);

	float new_rect_size = lamda * sqrt(last_rect[3] * last_rect[3] + last_rect[2] * last_rect[2]);
	float new_img_rect[] = { floor(img_center[1] - new_rect_size / 2), floor(img_center[0] - new_rect_size / 2), floor(new_rect_size), floor(new_rect_size) };

	if (new_img_rect[0] < 1)
	{
		new_img_rect[0] = 1;
	}
	if (new_img_rect[0] + new_img_rect[2] > im_color.cols)
	{
		new_img_rect[2] = im_color.cols - new_img_rect[0] - 1;
	}
	if (new_img_rect[1] < 1)
	{
		new_img_rect[1] = 1;
	}
	if (new_img_rect[1] + new_img_rect[3] > im_color.rows)
	{
		new_img_rect[3] = im_color.rows - new_img_rect[1] - 1;
	}

	Rect surround_rect(new_img_rect[0], new_img_rect[1], new_img_rect[2] + 1, new_img_rect[3] + 1);
	Mat surround_img;
	im_color(surround_rect).copyTo(surround_img);  // x <= pt.x < x + width, y <= pt.y < y + height

	float sz[] = { last_rect[2], last_rect[3] };
	float half_h = sz[0] / 4;
	float half_w = sz[1] / 4;

	float im_h = surround_img.rows;
	float im_w = surround_img.cols;

	int w_num = ceil(im_w / half_w);
	int h_num = ceil(im_h / half_h);

	vector<float> object_boxes;
	float tmp_box[] = { 0, 0, 0, 0 };
	float tmp_pos[] = { 0, 0 };
	for (int i = 1; i <= w_num; i = i + 1)
	{
		for (int j = 1; j <= h_num; j = j + 1)
		{
			if (i < w_num  && j < h_num)
			{
				tmp_pos[0] = j * half_h;
				tmp_pos[1] = i * half_w;

				tmp_box[0] = tmp_pos[1] - sz[1] / 2;
				tmp_box[1] = tmp_pos[0] - sz[0] / 2;
				tmp_box[2] = ceil(sz[1]);
				tmp_box[3] = ceil(sz[0]);

				tmp_box[0] = my_max(1 - tmp_box[2] / 2, my_min(im_w - tmp_box[2] / 2, tmp_box[0]));
				tmp_box[1] = my_max(1 - tmp_box[3] / 2, my_min(im_h - tmp_box[3] / 2, tmp_box[1]));
				tmp_box[0] = ceil(my_max(1, tmp_box[0]));
				tmp_box[1] = ceil(my_max(1, tmp_box[1]));

				tmp_box[0] = tmp_box[0] + new_img_rect[0];
				tmp_box[1] = tmp_box[1] + new_img_rect[1];

				if (tmp_box[0] < 1)
				{
					tmp_box[0] = 1;
				}
				if (tmp_box[0] + tmp_box[2] > im_color.cols)
				{
					tmp_box[0] = im_color.cols - tmp_box[2];
				}
				if (tmp_box[1] < 1)
				{
					tmp_box[1] = 1;
				}
				if (tmp_box[1] + tmp_box[3] > im_color.rows)
				{
					tmp_box[1] = im_color.rows - tmp_box[3];
				}

				object_boxes.push_back(tmp_box[0]);
				object_boxes.push_back(tmp_box[1]);
				object_boxes.push_back(tmp_box[2]);
				object_boxes.push_back(tmp_box[3]);
			}
		}
	}
	return object_boxes;
}

// tracker
void track()
{
	int i;
	string frameName;
	mwArray frame_no(1, 1, mxINT32_CLASS);
	mwArray x(1, 1, mxDOUBLE_CLASS);
	mwArray y(1, 1, mxDOUBLE_CLASS);
	mwArray w(1, 1, mxDOUBLE_CLASS);
	mwArray h(1, 1, mxDOUBLE_CLASS);
	mwArray new_x(1, 1, mxDOUBLE_CLASS);
	mwArray new_y(1, 1, mxDOUBLE_CLASS);
	mwArray new_w(1, 1, mxDOUBLE_CLASS);
	mwArray new_h(1, 1, mxDOUBLE_CLASS);

	double *xx = new double[1];
	double *yy = new double[1];
	double *ww = new double[1];
	double *hh = new double[1];
	double new_xx;
	double new_yy;
	double new_ww;
	double new_hh;

	const char * tmp_seq = video.c_str();
	mwArray seq(tmp_seq);

	frame_num = 0;

	while (true)
	{
		if (init_flag == 0)
		{
			continue;
		}
		if (verifyEnd && trackEnd)
		{
			// if verificator and tracker both ends, visualize the results and then exit
			for (int i = 1; i <= total_frames; i = i + 1)
			{
				stringstream strStream;
				strStream << "# " << i;
				putText(frame[i], strStream.str(), pt, CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0f, color);

				rectangle(frame[i], Point(tracking_results[i][1] - tracking_results[i][3] / 2, tracking_results[i][0] - tracking_results[i][2] / 2), Point(tracking_results[i][1] + tracking_results[i][3] / 2, tracking_results[i][0] + tracking_results[i][2] / 2), Scalar(0, 0, 255), 3, 8);
				imshow("Image", frame[i]);
				waitKey(1);
			}
			std::cout << "tracking finished ..." << std::endl;
			break;
		}
		if (new_start)
		{
			new_start = false;
			frame_num = new_frame;
			tracking_point = new_frame;
			//std::cout << "new tracking start point at frame: " << new_frame << std::endl;

			// update the tracker with a new tracking RoI
			new_xx = new_roi[0];
			new_yy = new_roi[1];
			new_ww = new_roi[2];
			new_hh = new_roi[3];

			new_x.SetData(&new_xx, 1);
			new_y.SetData(&new_yy, 1);
			new_w.SetData(&new_ww, 1);
			new_h.SetData(&new_hh, 1);

			// update tracker
			frame_no.SetData(&new_frame, 1);
			fDSST(4, x, y, w, h, frame_no, seq, new_x, new_y, new_w, new_h);

			x.GetData(xx, 1);
			y.GetData(yy, 1);
			w.GetData(ww, 1);
			h.GetData(hh, 1);

			tracking_results[new_frame].clear();
			// update tracking result
			tracking_results[new_frame].push_back(xx[0]);
			tracking_results[new_frame].push_back(yy[0]);
			tracking_results[new_frame].push_back(ww[0]);
			tracking_results[new_frame].push_back(hh[0]);

			trackChange = true;
			trackEnd = false;   // change the status of tracker if it ends
		}
		if (trackEnd)
		{
			// do nothing until verificator sends signal or tracker ends
			continue;
		}
		frame_num = frame_num + 1;
		frameName = get_image_name(base_path + video_path + "img\\", frame_num);

		frame[frame_num] = imread(frameName, CV_LOAD_IMAGE_COLOR);
		net_frame[frame_num] = imread(frameName, -1);

		// below is the tracking part
		new_xx = 0;
		new_yy = 0;
		new_ww = 0;
		new_hh = 0;

		frame_no.SetData(&frame_num, 1);
		new_x.SetData(&new_xx, 1);
		new_y.SetData(&new_yy, 1);
		new_w.SetData(&new_ww, 1);
		new_h.SetData(&new_hh, 1);

		std::cout << "processing frame " << frame_no << " ..." << std::endl;

		// The matlabKCF provides an interface to call fDSST.
		// We compile the matlab version fDSST, and obtain the .dll and .lib files. After that, we call it in C++
		// 
		fDSST(4, x, y, w, h, frame_no, seq, new_x, new_y, new_w, new_h);

		x.GetData(xx, 1);
		y.GetData(yy, 1);
		w.GetData(ww, 1);
		h.GetData(hh, 1);

		if (tracking_results[frame_num].size() > 0)
		{
			tracking_results[frame_num].clear();
		}
		tracking_results[frame_num].push_back(xx[0]);
		tracking_results[frame_num].push_back(yy[0]);
		tracking_results[frame_num].push_back(ww[0]);
		tracking_results[frame_num].push_back(hh[0]);

		tracking_point = tracking_point + 1;

		if (frame_num == end_num)
		{
			trackEnd = true;
			// break;
		}
	}
}

// verifies
void verify()
{
	vector<float> current_box;
	Mat curr_feat;
	Mat score;

	Classifier classifier = init_global_vars();

	init_flag = 1;

	while (true)
	{
		if (v_idx >= total_frames - V) // if remain frames are less than V, just stop
		{
			verifyEnd = true;
			break;
		}
		if (!trackChange)
		{
			continue;
		}
		if (v_idx + V < tracking_point)  // do verification every V frames
		{
			if (V == V_bk)
			{
				v_idx = int(v_idx / 10) * 10 + V;
			}
			else
			{
				v_idx = v_idx + V;
			}
			// verify the v_idx-th frame
			current_box.push_back(tracking_results[v_idx][1] - tracking_results[v_idx][3] / 2);
			current_box.push_back(tracking_results[v_idx][0] - tracking_results[v_idx][2] / 2);
			current_box.push_back(tracking_results[v_idx][3]);
			current_box.push_back(tracking_results[v_idx][2]);

			// get feature for tracking result in frame v_idx 
			curr_feat = get_network_feature(classifier, net_frame[v_idx], current_box);

			// compute verification score
			score = first_feat * curr_feat;  // score for current box

			// verifying score is smaller than the pre-defined threshold,
			// and we need to do object detection in a surrounding area
			if (score.at<float>(0, 0) < verify_th)
			{
				V = 1;

				vector<float> res_rect = current_box;     // rentangle in current box
				Mat im_color = net_frame[v_idx];          // current frame

				// results in current frame and last frame
				vector<float> curr_result;
				vector<float> last_result;
				curr_result.push_back(tracking_results[v_idx][0]);
				curr_result.push_back(tracking_results[v_idx][1]);
				curr_result.push_back(tracking_results[v_idx][2]);
				curr_result.push_back(tracking_results[v_idx][3]);
				last_result.push_back(tracking_results[v_idx - 1][0]);
				last_result.push_back(tracking_results[v_idx - 1][1]);
				last_result.push_back(tracking_results[v_idx - 1][2]);
				last_result.push_back(tracking_results[v_idx - 1][3]);

				// get candidates amd their features for detection
				vector<float> candidate_boxes = get_candidates(curr_result, last_result, im_color, lamda);
				Mat cand_feat = get_network_feature(classifier, im_color, candidate_boxes);

				// compute verification score for each candidate
				Mat candidate_scores = first_feat * cand_feat;  

				Mat score_idx;
				sortIdx(candidate_scores, score_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

				// get the candidate with highest score
				int max_id = score_idx.at<int>(0, 0);
				float max_score = candidate_scores.at<float>(0, max_id);

				// if detection result is reliable, then use it to correct tracker
				if (max_score >= det_threshold)
				{
					if (new_roi.size() > 0)
					{
						new_roi.clear();
					}
					new_roi.push_back(candidate_boxes[max_id * 4]);
					new_roi.push_back(candidate_boxes[max_id * 4 + 1]);
					new_roi.push_back(candidate_boxes[max_id * 4 + 2]);
					new_roi.push_back(candidate_boxes[max_id * 4 + 3]);

					// setup a new tracking point
					new_frame = v_idx;
					new_start = true;

					trackChange = false;
				}
			}
			else
			{
				V = V_bk;
			}
			current_box.clear();
		}
	}
}

// main function
int main(int argc, char* argv[])
{
	video = argv[1];

	if (fDSSTInitialize()){
		std::cout << "fDSST initialized successfully!" << std::endl;
	}
	else{
		std::cout << "fail to initialize fDSST!" << std::endl;
		return 0;
	}

	// create two threads: tracker and verifier
	std::thread th1(track);
	std::thread th2(verify);
	th1.join();
	th2.join();
	
	// kill fDSST tracker
	fDSSTTerminate();
	mclTerminateApplication();
	
	return 1;
}
