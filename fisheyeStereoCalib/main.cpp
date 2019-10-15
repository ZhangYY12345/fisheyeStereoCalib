#include "methods/fisheyeCalib3d.h"
#include "methods/fisheyeExpand.h"

using namespace cv;
using namespace std;

int main()
{
	std::string path_ = "D:\\studying\\stereo vision\\research code\\data\\20190924\\0924\\left\\fisheye\\1L.jpg";
	//fisheyeExpandApply(path_);
	Mat imgSrc = imread(path_);
	Mat imgDst;
	std::string resPath_pre = path_.substr(0, path_.length() - 4);

	fisheyeExpandTest(imgSrc, imgDst, STEREOGRAPHIC, resPath_pre + "_STEREOGRAPHIC");

	fisheyeExpandTest(imgSrc, imgDst, EQUIDISTANCE, resPath_pre + "_EQUIDISTANCE");

	fisheyeExpandTest(imgSrc, imgDst, EQUISOLID, resPath_pre + "_EQUISOLID");

	fisheyeExpandTest(imgSrc, imgDst, ORTHOGONAL, resPath_pre + "_ORTHOGONAL");

	waitKey();



	std::string imgFilePath_ = "D:\\studying\\stereo vision\\research code\\data\\2019-07-22";
	//std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-22\\stereoCalibData20190722.xml";
	std::string imgFilePath = "D:\\studying\\stereo vision\\research code\\data\\20190719\\camera_jpg_2";
	std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\20190719\\camera_jpg_2\\stereoCalibData20190720.xml";

	//std::string imgFilePath = "D:\\studying\\stereo vision\\research code\\data\\camera";
	//std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\camera\\stereoCalibrateData20190710.xml";

	//double singleRms = fisheyeCamCalibSingle(imgFilePath, xmlFilePath);

	double errCalib = stereoFisheyeCamCalib_3(imgFilePath, xmlFilePath);

	//rectify_(xmlFilePath, imgFilePath);

	//std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-08\\stereoCalibrateDataResult368_2.xml";
	Mat left_ = imread(imgFilePath_ + "\\testL.jpg");
	Mat right_ = imread(imgFilePath_ + "\\testR.jpg");
	//copyMakeBorder(left_, left_, 0, 0, 0, 8, BORDER_CONSTANT, Scalar(0, 0, 0));
	//imwrite("D:/1_.jpg", left_);
	//imwrite("D:/2_.jpg", right_);

	//Mat left_ = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedLeft.jpg");
	//Mat right_ = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedRight.jpg");



	Mat undistL, undistR;
	stereoFisheyeUndistort(left_, right_, xmlFilePath, undistL, undistR);

	return 0;
}