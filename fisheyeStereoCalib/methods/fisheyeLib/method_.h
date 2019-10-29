#pragma once

#include "parameterCalib_.h"

struct myCompare
{
	bool operator() (const cv::Point2d& node1, const cv::Point2d& node2) const
	{
		if(node1.x < node2.x)
		{
			return true;
		}
		if(node1.x == node2.x)
		{
			if (node1.y < node2.y)
				return true;
		}
		return false;
	}
}; 

//reading xml files
//void* xmlMapReader(xmlMapRead* file);

//fisheye calibration
void fisheyeCalib_(fisheyeCalibInfo infoStereoCalib);

//rectification
void fisheyeUndistort_(std::string filePath_, cv::Size imgSize, cv::Mat mapX, 
	cv::Mat mapY, std::vector<cv::Mat>& imgUndistort);

bool ptsDetect_calib(std::vector<cv::Mat> imgsL, std::vector<cv::Mat> imgsR, 
	douVecPt2f& ptsL, douVecPt2f& ptsR, douVecPt3f& ptsReal, int corRowNum, int corColNum);

void fisheyeCalcMap(std::string calibXml, std::map<cv::Point2d, std::vector<cv::Vec4d>, myCompare>& map2Dst, int& dstH, int& dstW);
void computeWeight(std::map<cv::Point2d, std::vector<cv::Vec4d>, myCompare>& mapOrigin, std::map<cv::Point2d, std::vector<cv::Vec3d>, myCompare>& mapDst);
void fisheyeRemap(cv::Mat src, cv::Mat& dst, std::map<cv::Point2d, std::vector<cv::Vec4d>, myCompare >& map2Dst, int dstH, int dstW);

void rectify_(calibInfo infoStereoCalib);
void saveFloatMat(cv::Mat& src, std::string filename);
void loadFloatMat(std::string filename, cv::Mat& dst);
