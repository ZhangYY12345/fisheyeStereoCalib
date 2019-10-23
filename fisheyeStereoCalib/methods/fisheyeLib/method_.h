#pragma once
#include "parameterCalib_.h"

//reading xml files
//void* xmlMapReader(xmlMapRead* file);

//fisheye calibration
void fisheyeCalib_(fisheyeCalibInfo infoStereoCalib);

//rectification
void fisheyeUndistort_(std::string filePath_, cv::Size imgSize, cv::Mat mapX, 
	cv::Mat mapY, std::vector<cv::Mat>& imgUndistort);

bool ptsDetect_calib(std::vector<cv::Mat> imgsL, std::vector<cv::Mat> imgsR, 
	douVecPt2f& ptsL, douVecPt2f& ptsR, douVecPt3f& ptsReal, int corRowNum, int corColNum);

void fisheyeCalcMap(std::string calibXml, cv::Mat& mapx_ceil, cv::Mat& mapx_floor, cv::Mat& mapy_ceil, cv::Mat& mapy_floor);
void fisheyeRemap(cv::Mat src, cv::Mat& dst, cv::Mat& mapx_ceil, cv::Mat& mapx_floor, cv::Mat& mapy_ceil, cv::Mat& mapy_floor);

//void rectify_(calibInfo infoStereoCalib);
