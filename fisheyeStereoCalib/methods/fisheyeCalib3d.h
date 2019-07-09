#pragma once
#include <opencv2/opencv.hpp>
#include <string>

//single camera calibreation
void myCameraCalibration(std::string cameraParaPath);
void myCameraCalibration(std::string imgFilePath, std::string cameraParaPath);
void myCameraUndistort(std::string cameraParaPath);
void myCameraUndistort(std::string imgFilePath, std::string cameraParaPath);

//two camera calibration and stereo vision
void twoCamerasCalibration(std::string cameraParaPath);
void twoCamerasCalibration(std::string imgFilePath, std::string cameraParaPath);
void twoCamerasCalibration(std::string imgFilePathL, std::string imgFilePathR, std::string cameraParaPath);

cv::Mat mergeRectification(const cv::Mat& l, const cv::Mat& r);
void stereoFisheyeCamCalib(std::string imgFilePathL, std::string imgFilePathR, std::string cameraParaPath);
void stereoFisheyCamCalibRecti(std::string imgFilePathL, std::string cameraParaPath);


void stereoCameraUndistort(std::string cameraParaPath);
void stereoCameraUndistort(std::string imgFilePath, std::string cameraParaPath);
void getRectifiedImages(std::string imgFilePath, std::string cameraParaPath);
void getRectifiedImages(cv::Mat imgLeft, cv::Mat imgRight, std::string cameraParaPath,
	cv::Mat& rectifiedLeft, cv::Mat& rectifiedRight);
