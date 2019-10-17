#pragma once
#include <opencv2/opencv.hpp>
#include "corrector.h"

//apply the class to expand the fisheye image
void fisheyeExpand(cv::Mat src, cv::Mat& dst, bool isLeft = true);
void fisheyeExpand(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, corrector::correctMethod method_, bool isLeft = true);

void fisheyeExpandTest(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, std::string resPathPre, bool isLeft = true);
void fisheyeExpandApply(std::string imgPath, bool isLeft = true);