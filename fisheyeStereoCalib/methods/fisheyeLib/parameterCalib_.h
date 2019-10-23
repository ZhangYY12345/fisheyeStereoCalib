#pragma once

#include "Calibration.h"
#include "LineDetection.h"
#include "Pair.h"
#include "Pattern.h"

typedef  std::vector<std::vector<cv::Point2f> >  douVecPt2f;
typedef  std::vector<std::vector<cv::Point3f> >  douVecPt3f;

struct xmlMapRead
{
	std::string xmlMapName;
	cv::Mat mapImg;
};
struct fisheyeCalibInfo
{
	std::string calibPatternFile;	//fisheye camera calibration images: stripe images' path
									//xml file to store pattern images' path	
	std::string calibLineDetected;	//xml file to store information of detected lines
	std::string calibFile;			//xml file to store fisheye camera calibration data using the fisheye library
};

struct calibInfo
{
	std::string calibFileL;			//"resCalibL.xml",left fisheye camera calibration data using the fisheye library
	std::string calibFileR;			//"resCalibR.xml",right fisheye camera calibration data using the fisheye library

	std::string calibChessImgPathL;	//chessboard images from left fisheye camera for stereo calibration: undistort these images and detect cross points inside for later stereo calibration using cv::stereoCalibration()
	std::string calibChessImgPathR;	//chessboard images from right fisheye camera for stereo calibration
	int chessRowNum;
	int chessColNum;
	std::string stereoCalib;		//stereo calibration data based on rectified chessboard images
	std::string stereoCalib_undistort_mapxL;
	std::string stereoCalib_undistort_mapyL;
	std::string stereoCalib_undistort_mapxR;
	std::string stereoCalib_undistort_mapyR;
	std::string stereoCalib_rectify_mapxL;
	std::string stereoCalib_rectify_mapyL;
	std::string stereoCalib_rectify_mapxR;
	std::string stereoCalib_rectify_mapyR;
};