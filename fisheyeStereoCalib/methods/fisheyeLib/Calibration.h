//
//  Calibration.h
//  Calibration
//
//  Created by Ryohei Suda on 2014/09/11.
//  Copyright (c) 2014年 RyoheiSuda. All rights reserved.
//

#ifndef __Calibration__Calibration__
#define __Calibration__Calibration__

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
//#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "./calib_libs/IncidentVector.h"
#include "./calib_libs/EquidistanceProjection.h"			//等距投影
#include "./calib_libs/EquisolidAngleProjection.h"		//等立体角投影
#include "./calib_libs/StereographicProjection.h"		//体视投影
#include "./calib_libs/OrthographicProjection.h"			//正交投影
#include "Pair.h"
#include "./calib_libs/tinyxml2.h"

class Calibration {
  
public:
    //TODO Remove overlap variables with IncidentVecotr.h
//    cv::Point2d center;
//    cv::Size2i img_size;
//    double f; // focal length is pixel unit
//    double const f0 = 150; // Scale constant;
//    std::vector<double> a;
    std::vector<Pair> edges;
    
    void setParameters(std::vector<Pair>& edges, double& f, double& f0, cv::Point2d& center, cv::Size2i& img_size, int a_size);
    void loadData(std::string);
    void save(std::string);
    
    void calibrate(bool divide); // Do calibrate
    void calibrate2(); // For new optimazation test
    
    
    //大文字の後に付いてるcは微分を表している
    double J1(); // Colinearity:共线性约束
    double J1c(int);
    double J1cc(int, int);
    double J2(); // Parallelism：平行性约束
    double J2c(int);
    double J2cc(int, int);
    double J3(); // Orthogonality：正交性约束
    double J3c(int);
    double J3cc(int, int);
};

#endif /* defined(__Calibration__Calibration__) */
