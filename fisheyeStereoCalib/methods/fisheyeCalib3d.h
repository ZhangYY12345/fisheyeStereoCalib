#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "parametersStereo.h"

void createCalibPicCircle(int width = 2560, int height = 1440, int numX = 30, int numY = 18);
void createCalibPicSquare(int width = 2560, int height = 1440, int numX = 30, int numY = 18);
void createStripePic_withSquare(int width = 2560, int height = 1440, int numX = 30, int numY = 18, int slope = 5);

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


//detect pts in the pre-stereo_calibration step
bool ptsCalib(std::string imgFilePathL, cv::Size& imgSize,
	douVecPt2f& ptsL, douVecPt2f& ptsR, douVecPt3f& ptsReal,
	int corRowNum, int corColNum);
bool ptsCalib(std::vector<cv::Mat> imgsL, std::vector<cv::Mat> imgsR, douVecPt2f& ptsL, douVecPt2f& ptsR,
	douVecPt3f& ptsReal, int corRowNum, int corColNum);

bool ptsCalib_Single(std::string imgFilePath, cv::Size& imgSize, douVecPt2f& pts,
	douVecPt3f& ptsReal, int corRowNum, int corColNum);
bool ptsCalib_Single(std::vector<cv::Mat> imgs, douVecPt2f& pts,
	douVecPt3f& ptsReal, int corRowNum, int corColNum);

//---------------------------------------------------
struct myCmp_map
{
	bool operator() (const cv::Point2i& node1, const cv::Point2i& node2) const
	{
		if (node1.x < node2.x)
		{
			return true;
		}
		if (node1.x == node2.x)
		{
			if (node1.y < node2.y)
				return true;
		}
		return false;
	}
};

void createMask_lines(cv::Mat& dst);
cv::Mat detectLines_(cv::Mat& src1, cv::Mat& src2, bool isHorizon);
void detectLines_(cv::Mat src1, cv::Mat src2, cv::Mat& dst, cv::Mat& dst_inv, bool isHorizon);
void connectEdge(cv::Mat& src, int winSize_thres, bool isHorizon = true);
void connectEdge2(cv::Mat& src, int winSize_thres, bool isHorizon = true);

void myGetLines(cv::Mat& src, cv::Mat& tmp, cv::Point2i startPt, std::vector<cv::Point2i>& oneLine, int lenThres, bool isHorizon = true);
void removeShortEdges(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon = true, RIGHT_COUNT_SIDE mode = TOP_LEFT);
void post_process(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, bool isHorizon = true, RIGHT_COUNT_SIDE mode = TOP_LEFT);
void detectPts(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal, double grid_size);
void detectPts(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal, 
	double grid_size, int hNum, int vNum, RIGHT_COUNT_SIDE mode, cv::Mat mask = cv::Mat::Mat());
void detectPts2(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal,
	double grid_size, int hNum, int vNum, RIGHT_COUNT_SIDE mode, cv::Mat mask = cv::Mat::Mat());

void loadXML_imgPath(std::string xmlPath, cv::Size& imgSize, std::map<RIGHT_COUNT_SIDE, std::vector<std::vector<std::string> > >& path_);
bool ptsCalib_single2(std::string xmlFilePath, cv::Size& imgSize,
	douVecPt2f& pts, douVecPt3f& ptsReal, double gridSize, int hNum, int vNum, cv::Mat mask = cv::Mat::Mat());


//----------------------------------------------------
double stereoCamCalibration(std::string imgFilePath, std::string cameraParaPath);
double stereoCamCalibration_2(std::string imgFilePathL, std::string cameraParaPath);

//undistort image/image rectification




/***********************************************************
 *************  fisheye camera calibration   ***************
 ***********************************************************
*/
//single fisheye camera calibration
double fisheyeCamCalibSingle(std::string imgFilePath, std::string cameraParaPath);
void distortRectify_fisheye(cv::Mat K, cv::Mat D, cv::Size imgSize, std::string imgFilePath);
void distortRectify_fisheye(cv::Mat K, cv::Mat D, cv::Size imgSize, std::string imgFilePath, 
	std::vector<cv::Mat>& undistortImgs, bool isLeft = true);
//stereo fisheye calibration
double stereoFisheyeCamCalib(std::string imgFilePathL, std::string cameraParaPath);
double stereoFisheyeCamCalib_2(std::string imgFilePathL, std::string cameraParaPath);
double stereoFisheyeCamCalib_3(std::string imgFilePathL, std::string cameraParaPath);
//fisheye image undistort
void merge4();
void stereoFisheyeUndistort(cv::Mat distLeft, cv::Mat distRight, std::string cameraParaPath, cv::Mat& rectiLeft, cv::Mat& rectiRight);
void rectify_(std::string cameraParaPath, std::string imgPath);

/***********************************************************
 *****************  image rectification   ******************
 ***********************************************************
*/
cv::Mat mergeRectification(const cv::Mat& l, const cv::Mat& r);
void stereoCameraUndistort(std::string imgFilePath, std::string cameraParaPath);
void getRectifiedImages(std::string imgFilePath, std::string cameraParaPath);
void getRectifiedImages(cv::Mat imgLeft, cv::Mat imgRight, std::string cameraParaPath,
	cv::Mat& rectifiedLeft, cv::Mat& rectifiedRight);
