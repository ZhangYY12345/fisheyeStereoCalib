#include "methods/fisheyeCalib3d.h"

using namespace cv;
using namespace std;

int main()
{
	std::string imgFilePath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-08";
	std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-08\\stereoCalibrateData20190710.xml";

	//std::string imgFilePath = "D:\\studying\\stereo vision\\research code\\data\\camera";
	//std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\camera\\stereoCalibrateData20190710.xml";

	double errCalib = stereoCamCalibration(imgFilePath, xmlFilePath);

	//std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-08\\stereoCalibrateDataResult368_2.xml";
	Mat left_ = imread("D:\\imgs20190627\\left_1.jpeg");
	Mat right_ = imread("D:\\imgs20190627\\right_1.jpeg");
	//copyMakeBorder(left_, left_, 0, 0, 0, 8, BORDER_CONSTANT, Scalar(0, 0, 0));
	//imwrite("D:/1_.jpg", left_);
	//imwrite("D:/2_.jpg", right_);

	//Mat left_ = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedLeft.jpg");
	//Mat right_ = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedRight.jpg");


	Mat undistL, undistR;
	stereoFisheyeUndistort(left_, right_, xmlFilePath, undistL, undistR);

	return 0;
}