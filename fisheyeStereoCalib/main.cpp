#include "methods/fisheyeCalib3d.h"
#include "methods/fisheyeExpand.h"
#include "methods/fisheyeLib/method_.h"

using namespace cv;
using namespace std;

int main()
{
	/*
	//single fisheye camera calibration
	fisheyeCalibInfo calibInfoL, calibInfoR;
	calibInfoL.calibPatternFile = "./20191017-1-2/patternsL.xml";
	calibInfoL.calibLineDetected = "./20191017-1-2/linesDetectedL.xml";
	calibInfoL.calibFile = "./20191017-1-2/resCalibL.xml";
	fisheyeCalib_(calibInfoL);

	calibInfoR.calibPatternFile = "./20191017-1-2/patternsR.xml";
	calibInfoR.calibLineDetected = "./20191017-1-2/linesDetectedR.xml";
	calibInfoR.calibFile = "./20191017-1-2/resCalibR.xml";
	fisheyeCalib_(calibInfoR); 

	waitKey();*/
	/*
	calibInfo infoCalib;
	infoCalib.calibFileL = "./20191017-1-2/resCalibL.xml";
	infoCalib.calibFileR = "./20191017-1-2/resCalibR.xml";
	infoCalib.fisheye_reprojectL = "./20191017-1-2/resReprojectL.xml";
	infoCalib.fisheye_reprojectR = "./20191017-1-2/resReprojectR.xml";

	infoCalib.calibChessImgPathL = "D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\left";
	infoCalib.calibChessImgPathR = "D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\right";
	infoCalib.chessRowNum = 6;
	infoCalib.chessColNum = 9;
	infoCalib.stereoCalib = "./20191017-1-2/unditortStereoCalib.xml";
	infoCalib.stereoCalib_undistort_mapxL = "./20191017-1-2/undistort_mapxL.dat";
	infoCalib.stereoCalib_undistort_mapyL = "./20191017-1-2/undistort_mapyL.dat";
	infoCalib.stereoCalib_undistort_mapxR = "./20191017-1-2/undistort_mapxR.dat";
	infoCalib.stereoCalib_undistort_mapyR = "./20191017-1-2/undistort_mapyR.dat";

	infoCalib.stereoCalib_rectify_mapxL = "./20191017-1-2/rectify_mapxL.xml";
	infoCalib.stereoCalib_rectify_mapyL = "./20191017-1-2/rectify_mapyL.xml";
	infoCalib.stereoCalib_rectify_mapxR = "./20191017-1-2/rectify_mapxR.xml";
	infoCalib.stereoCalib_rectify_mapyR = "./20191017-1-2/rectify_mapyR.xml";

	//rectify_(infoCalib);*/

	//std::string path_ = "D:\\studying\\stereo vision\\research code\\data\\20191017-2\\left\\1L.jpg";
	std::string path__ = "D:\\studying\\stereo vision\\research code\\data\\20191017-3\\left\\real_fisheye_stereoGraphic\\ideal_stereographic\\left_undistort.jpg";

	//fisheyeExpandApply(path_);
	Mat imgSrc = imread(path__);
	cv::Mat imgDistort;


	Mat imgDst;
	std::string resPath_pre = path__.substr(0, path__.length() - 4);
	// for distorted fisheye images
	// for left camera:
	// center = cv::Point(1283, 724)
	// radius = 1199
	// for right camera
	// center = cv::Point(1283, 722)
	// radius = 1187


	// for undistorted fisheye images
	// for ideal equidistance model and ideal equisolidAngle model
	// center = cv::Point(1560, 940)
	// radius = 1286

	// for ideal stereographic model
	// center = cv::Point(2560, 1440)
	// radius = 2270
	cv::Point2i center = cv::Point(2560, 1440);
	int radius = 2270;

	fisheyeExpandTest(imgSrc, imgDst, STEREOGRAPHIC, resPath_pre + "_STEREOGRAPHIC", center, radius);

	//fisheyeExpandTest(imgSrc, imgDst, EQUIDISTANCE, resPath_pre + "_EQUIDISTANCE", center, radius);

	//fisheyeExpandTest(imgSrc, imgDst, EQUISOLID, resPath_pre + "_EQUISOLID", center, radius);

	//fisheyeExpandTest(imgSrc, imgDst, ORTHOGONAL, resPath_pre + "_ORTHOGONAL", center, radius);

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