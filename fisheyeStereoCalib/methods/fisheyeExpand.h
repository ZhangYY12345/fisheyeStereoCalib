#pragma once
#include <opencv2/opencv.hpp>
#include "corrector.h"

//apply the class to expand the fisheye image
void fisheyeExpand(cv::Mat src, cv::Mat& dst, cv::Point2i center, int radius);
void fisheyeExpand(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, corrector::correctMethod method_, cv::Point2i center, int radius);

void fisheyeExpandTest(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, std::string resPathPre, cv::Point2i center, int radius);
void fisheyeExpandApply(std::string imgPath, cv::Point2i center, int radius);