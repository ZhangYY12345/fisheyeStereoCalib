#pragma once
#include <opencv2/opencv.hpp>
#include "corrector.h"

//apply the class to expand the fisheye image
void fisheyeExpand(cv::Mat src, cv::Mat& dst);
void fisheyeExpand(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, corrector::correctMethod method_);

void fisheyeExpandTest(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, std::string resPathPre);
void fisheyeExpandApply(std::string imgPath);