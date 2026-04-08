#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <windows.h>
#include <time.h>
#include <utility>
#include <vector>

using namespace caffe;

class Classifier
{
public:
	Classifier(const string& model_file, const string& trained_file, const string& mean_file);
	std::vector<float> Classify(const cv::Mat& img, int img_width, int img_height, int img_sz, vector<float> boxes);
private:
	void SetMean(const string& mean_file);
	std::vector<float> Predict(const cv::Mat& img, int img_width, int img_height, int img_sz, vector<float> boxes);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
};

Classifier::Classifier(const string& model_file, const string& trained_file, const string& mean_file)
{
	Caffe::set_mode(Caffe::GPU);
	// Load the network. 
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	// Load the binaryproto mean file. 
	SetMean(mean_file);
}


std::vector<float> Classifier::Classify(const cv::Mat& img, int img_width, int img_height, int img_sz, vector<float> boxes)
{
	std::vector<float> output = Predict(img, img_width, img_height, img_sz, boxes);
	int box_num = boxes.size() / 4;
	cv::Mat result_mat(output);

	return output;
}

void Classifier::SetMean(const string& mean_file)
{
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	// Convert from BlobProto to Blob<float> 
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	cv::Mat mean;
	cv::merge(channels, mean);

	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img, int img_width, int img_height, int img_sz, vector<float> boxes) {
	int box_num = boxes.size()/4;
	cv::Mat boxes_mat(boxes);
	cv::Mat new_boxes_mat;
	new_boxes_mat = boxes_mat.reshape(0, box_num);

	cv::Mat s_mat = cv::Mat::zeros(new_boxes_mat.rows, 5, CV_32F); // boxes input into the network
	for (int i = 0; i < new_boxes_mat.rows; i = i + 1)
	{
		s_mat.at<float>(i, 1) = new_boxes_mat.at<float>(i, 0) * img_sz / img_width;
		s_mat.at<float>(i, 2) = new_boxes_mat.at<float>(i, 1) * img_sz / img_height;
		s_mat.at<float>(i, 3) = (new_boxes_mat.at<float>(i, 0) + new_boxes_mat.at<float>(i, 2) - 1) * img_sz / img_width;
		s_mat.at<float>(i, 4) = (new_boxes_mat.at<float>(i, 3) + new_boxes_mat.at<float>(i, 1) - 1) * img_sz / img_height;
	}
	s_mat.colRange(1, s_mat.cols) = s_mat.colRange(1, s_mat.cols) - 1;

	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, img_sz, img_sz);

	Blob<float>* roi_input_layer = net_->input_blobs()[1];
	roi_input_layer->Reshape(box_num, 5, 1, 1);

	// Forward dimension change to all layers. 
	net_->Reshape();

	int roi_layer = net_->num_inputs();
	// std::cout << "number of inputs " << roi_layer << std::endl;

	int num = net_->blob_by_name("rois")->num();
	// std::cout << "number of rois " << num << std::endl;

	int output_num = net_->blob_by_name("feat_l2")->num();
	// std::cout << "number of outputs " << output_num << std::endl;

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	// net_->blob_by_name("rois")->set_cpu_data(input_roi[0]);
	float* myptr = (float*)s_mat.data;
	net_->blob_by_name("rois")->set_cpu_data(myptr);

	net_->Forward();

	// Copy the output layer to a std::vector 
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	// const float* end = begin + output_layer->channels();
	const float* end = begin + output_layer->count();
	return std::vector<float>(begin, end);
}

void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
	// Convert the input image to the input image format of the network. 
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}