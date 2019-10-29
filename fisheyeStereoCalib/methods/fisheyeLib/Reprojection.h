//
//  Reprojection.h
//  Reprojection
//
//  Created by Ryohei Suda on 2014/09/12.
//  Copyright (c) 2014年 RyoheiSuda. All rights reserved.
//

#ifndef __Reprojection__Reprojection__
#define __Reprojection__Reprojection__

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "./calib_libs/IncidentVector.h"
#include "./calib_libs/OrthographicProjection.h"
#include "./calib_libs/StereographicProjection.h"
#include "./calib_libs/EquisolidAngleProjection.h"
#include "./calib_libs/EquidistanceProjection.h"

#define M_PI 3.1415926

class Reprojection {
public:
    int precision = 100; //?
    std::vector<double> t2r; // theta to radius
    std::vector<double> r2t;
    double rad_step;
	double ideal_r;
	cv::Point2d ideal_center;
    std::string projection;
    
    void loadParameters(std::string);
    void theta2radius();
    void saveTheta2Radius(std::string filename);
	void loadTheta2Radius(std::string filename);
    void saveRadius2Theta(std::string filename);
	void loadRadius2Theta(std::string filename);
	void saveReprojectData(std::string fileName, bool isSave_t2r);
    void calcMaps(double theta_x, double theta_y, double f_, cv::Mat& mapx, cv::Mat& mapy);
	void calcMaps(double f_, cv::Mat& mapx, cv::Mat& mapy);
	void calcMaps_fisheye_model_full(double theta_x, double theta_y, double f_, cv::Mat& mapx, cv::Mat& mapy, int scale = 2, bool isFisheyeModel = true);
	void calcMaps_fisheye_model_full(double f_, cv::Mat& mapx, cv::Mat& mapy, int scale = 2, bool isFisheyeModel = true);
	void calcMaps_fisheye_model_offset_full(double theta_x, double theta_y, double f_, cv::Mat& mapx, cv::Mat& mapy, int offset, bool isFisheyeModel = true);
	void calcMaps_fisheye_model_offset_full(double f_, cv::Mat& mapx, cv::Mat& mapy, int offset, bool isFisheyeModel = true);
};


#endif /* defined(__Reprojection__Reprojection__) */
