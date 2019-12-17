#include "methods/fisheyeCalib3d.h"
#include "methods/fisheyeExpand.h"
#include "methods/fisheyeLib/method_.h"

using namespace cv;
using namespace std;

int main()
{
	/*
	vector<cv::Mat> imgs;
	cv::Mat img0 = imread("D:/studying/stereo vision/research code/data/20191017-1/left/patternsImgL/3_pattern0.jpeg");
	cvtColor(img0, img0, COLOR_BGR2GRAY);
	cv::Mat img1 = imread("D:/studying/stereo vision/research code/data/20191017-1/left/patternsImgL/3_pattern1.jpeg");
	cvtColor(img1, img1, COLOR_BGR2GRAY);
	cv::Mat img2 = imread("D:/studying/stereo vision/research code/data/20191017-1/left/patternsImgL/3_pattern2.jpeg");
	cvtColor(img2, img2, COLOR_BGR2GRAY);
	cv::Mat img3 = imread("D:/studying/stereo vision/research code/data/20191017-1/left/patternsImgL/3_pattern3.jpeg");
	cvtColor(img3, img3, COLOR_BGR2GRAY);
	imgs.push_back(img0);
	imgs.push_back(img1);
	imgs.push_back(img2);
	imgs.push_back(img3);

	int grid_Size = 20;
	int rowNum = 9;
	int colNum = 16;
	vector<cv::Point2d> ptsImg;
	vector<cv::Point3d> ptsObj;
	detectPts(imgs, ptsImg, ptsObj, grid_Size, rowNum, colNum, TOP_LEFT);
	*/

	//fisheyeModel_show2(IDEAL_PERSPECTIVE);
	//createStripePic_withSquare();
	//createCalibPicSquare();

	/*
	//single fisheye camera calibration
	fisheyeCalibInfo calibInfoL, calibInfoR;
	calibInfoL.calibPatternFile = "patternsL_equalSolid.xml";//20191211
	calibInfoL.calibLineDetected = "./20191211/linesDetectedL_equalSolid.xml";
	calibInfoL.calibFile = "./20191211/resCalibL_equalSolid.xml";
	fisheyeCalib_(calibInfoL);

	calibInfoR.calibPatternFile = "./20191119/patternsR.xml";
	calibInfoR.calibLineDetected = "./20191119/linesDetectedR.xml";
	calibInfoR.calibFile = "./20191119/resCalibR.xml";
	fisheyeCalib_(calibInfoR); 
	*/
	//waitKey();
	/*
	calibInfo infoCalib;
	infoCalib.calibFileL = "./20191119/resCalibL.xml";
	infoCalib.calibFileR = "./20191119/resCalibR.xml";
	infoCalib.fisheye_reprojectL = "./20191119/resReprojectL.xml";
	infoCalib.fisheye_reprojectR = "./20191119/resReprojectR.xml";

	infoCalib.calibChessImgPathL = "D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\left";
	infoCalib.calibChessImgPathR = "D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\right";
	infoCalib.chessRowNum = 6;
	infoCalib.chessColNum = 9;
	infoCalib.stereoCalib = "./20191119/unditortStereoCalib.xml";
	infoCalib.stereoCalib_undistort_mapxL = "./20191119/undistort_mapxL.dat";
	infoCalib.stereoCalib_undistort_mapyL = "./20191119/undistort_mapyL.dat";
	infoCalib.stereoCalib_undistort_mapxR = "./20191119/undistort_mapxR.dat";
	infoCalib.stereoCalib_undistort_mapyR = "./20191119/undistort_mapyR.dat";

	infoCalib.stereoCalib_rectify_mapxL = "./20191119/rectify_mapxL.xml";
	infoCalib.stereoCalib_rectify_mapyL = "./20191119/rectify_mapyL.xml";
	infoCalib.stereoCalib_rectify_mapxR = "./20191119/rectify_mapxR.xml";
	infoCalib.stereoCalib_rectify_mapyR = "./20191119/rectify_mapyR.xml";

	rectify_(infoCalib);
	return 0;*/
	/*
	//std::string path_ = "D:\\studying\\stereo vision\\research code\\data\\20191017-2\\left\\1L.jpg";
	std::string path__ = "D:\\studying\\stereo vision\\research code\\data\\20191017-3\\left\\real_fisheye_stereoGraphic\\ideal_stereographic\\left_undistort.jpg";

	//fisheyeExpandApply(path_);
	cv::Mat imgSrc = imread(path__);
	cv::Mat imgDistort;


	cv::Mat imgDst;
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

	waitKey();*/



	std::string imgFilePath_ = "D:\\studying\\stereo vision\\research code\\data\\2019-07-22";
	//std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-22\\stereoCalibData20190722.xml";
	std::string imgFilePath = "D:/studying/stereo vision/research code/data/20190924/0924/left/fisheye";///img_border
	std::string xmlFilePath = "D:/studying/stereo vision/research code/data/20190924/0924/left/fisheye/stereoCalibData20190924.xml";///img_border

	//std::string imgFilePath = "D:\\studying\\stereo vision\\research code\\data\\camera";
	//std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\camera\\stereoCalibrateData20190710.xml";

	std::string xmlpath_img = "D:/studying/stereo vision/research code/fisheye-stereo-calibrate/fisheyeStereoCalib/fisheyeStereoCalib/fisheyeStereoCalib/patternsL20191211.xml";
	std::string xmlFile_path = "D:/studying/stereo vision/research code/fisheye-stereo-calibrate/fisheyeStereoCalib/fisheyeStereoCalib/fisheyeStereoCalib/20191211res_r_d.xml";

	double singleRms = fisheyeCamCalibSingle(xmlpath_img, xmlFilePath);

	cv::Mat K, D;
	cv::Size imgSize;
	FileStorage fn(xmlFilePath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["CameraInnerPara"] >> K;
	fn["CameraDistPara"] >> D;
	fn.release();

	string to_undistort = "D:/studying/stereo vision/research code/data/20191017-3/left";
	distortRectify_fisheye(K, D, imgSize, to_undistort);


	//double errCalib = stereoFisheyeCamCalib_3(imgFilePath, xmlFilePath);

	//rectify_(xmlFilePath, imgFilePath);

	//std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-08\\stereoCalibrateDataResult368_2.xml";
	cv::Mat left_ = imread(imgFilePath_ + "\\testL.jpg");
	cv::Mat right_ = imread(imgFilePath_ + "\\testR.jpg");
	//copyMakeBorder(left_, left_, 0, 0, 0, 8, BORDER_CONSTANT, Scalar(0, 0, 0));
	//imwrite("D:/1_.jpg", left_);
	//imwrite("D:/2_.jpg", right_);

	//cv::Mat left_ = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedLeft.jpg");
	//cv::Mat right_ = imread("D:\\studying\\stereo vision\\research code\\data\\CALIB20190702\\rectifiedRight.jpg");



	cv::Mat undistL, undistR;
	stereoFisheyeUndistort(left_, right_, xmlFilePath, undistL, undistR);

	return 0;
}