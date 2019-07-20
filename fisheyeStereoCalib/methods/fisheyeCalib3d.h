#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "parametersStereo.h"

void createCalibPicCircle(int width = 2560, int height = 1440, int numX = 30, int numY = 18);
void createCalibPicSquare(int width = 2560, int height = 1440, int numX = 30, int numY = 18);

/***********************************************************
 *************  general camera calibration   ***************
 ***********************************************************
*/
//single camera calibreation
void myCameraCalibration(std::string cameraParaPath);
void myCameraCalibration(std::string imgFilePath, std::string cameraParaPath);
void myCameraUndistort(std::string cameraParaPath);
void myCameraUndistort(std::string imgFilePath, std::string cameraParaPath);

//stereo calibration
double stereoCamCalibration(std::string cameraParaPath);
//detect pts in the pre-stereo calibration step
bool ptsCalib(std::string imgFilePathL, cv::Size& imgSize, douVecPt2f& ptsL, douVecPt2f& ptsR, douVecPt3f& ptsReal, int corRowNum, int corColNum);
bool ptsCalibSingle(std::string imgFilePath, cv::Size& imgSize, douVecPt2f& pts, douVecPt3f& ptsReal, int corRowNum, int corColNum);
//
double stereoCamCalibration(std::string imgFilePath, std::string cameraParaPath);
double stereoCamCalibration_2(std::string imgFilePathL, std::string cameraParaPath);

//undistort image/image rectification




/***********************************************************
 *************  fisheye camera calibration   ***************
 ***********************************************************
*/
double fisheyeCamCalibSingle(std::string imgFilePath, std::string cameraParaPath);
void distortRectify_fisheye(cv::Mat K, cv::Mat D, cv::Size imgSize, std::string imgFilePath);
double stereoFisheyeCamCalib(std::string imgFilePathL, std::string cameraParaPath);
double stereoFisheyeCamCalib_2(std::string imgFilePathL, std::string cameraParaPath);
double stereoFisheyCamCalib_3(std::string imgFilePathL, std::string cameraParaPath);

void merge4();
void stereoFisheyeUndistort(cv::Mat distLeft, cv::Mat distRight, std::string cameraParaPath, cv::Mat& rectiLeft, cv::Mat& rectiRight);


/***********************************************************
 *****************  image rectification   ******************
 ***********************************************************
*/
cv::Mat mergeRectification(const cv::Mat& l, const cv::Mat& r);
void stereoCameraUndistort(std::string imgFilePath, std::string cameraParaPath);
void getRectifiedImages(std::string imgFilePath, std::string cameraParaPath);
void getRectifiedImages(cv::Mat imgLeft, cv::Mat imgRight, std::string cameraParaPath,
	cv::Mat& rectifiedLeft, cv::Mat& rectifiedRight);
