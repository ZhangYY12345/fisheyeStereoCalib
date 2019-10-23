//
//  incidentVector.cpp
//  Calibration
//
//  Created by Ryohei Suda on 2014/03/31.
//  Copyright (c) 2014 Ryohei Suda. All rights reserved.
//

#include "IncidentVector.h"

//?全局变量?
double IncidentVector::f, IncidentVector::f0;
std::vector<double> IncidentVector::a;
std::vector<double> IncidentVector::b;
cv::Point2d IncidentVector::center;
cv::Size2i IncidentVector::img_size;
int IncidentVector::nparam = 3;
std::string IncidentVector::projection_name[PROJECTION_NUM] = {
	"Stereographic", "Orthographic", "Equidistance", "EquisolidAngle"};
int IncidentVector::projection;

//标定角点在图像中的图像物理坐标
IncidentVector::IncidentVector(cv::Point2d p)
{
    point = p;
}

//设置相机内参，
void IncidentVector::setParameters(double f, double f0, std::vector<double> a, 
	cv::Size2i img_size, cv::Point2d center)
{
    IncidentVector::f = f;
    IncidentVector::f0 = f0;
    IncidentVector::a = a;		//相机畸变参数a_1,a_2,a_3,...
    IncidentVector::img_size = img_size;
    IncidentVector::center = center;
    
    nparam = 3 + (int)a.size();
}

//projection indicate the chosen projection model of the fisheye camera :
//stereographic or orthographic or equidistance or equisolid_angle
void IncidentVector::setProjection(std::string projection)
{
    for (int i = 0; i < PROJECTION_NUM; ++i) {
        if (projection_name[i] == projection) {
            IncidentVector::projection = i;
            return;
        }
    }
    exit(4);
}

//calculate the coordinate(x_c,y_c,z_c) in camera coordinate system(Oc-XcYcZc)
void IncidentVector::calcM()
{
    r = sqrt(pow(center.x-point.x, 2) + pow(center.y-point.y, 2));	//r,是否考虑畸变？
    theta = aoi(r);
    if (r != 0) {
        m.x = ((point.x - center.x) / r) * sin(theta);
        m.y = ((point.y - center.y) / r) * sin(theta);
        m.z = cos(theta);
    } else {
        m.x = 0;
        m.y = 0;
        m.z = 1;
    }
}

// 各坐标关于theta求导
void IncidentVector::calcCommonPart()
{
    if (r != 0) {
        part.x = (point.x - center.x)/r * cos(theta);
        part.y = (point.y - center.y)/r * cos(theta);
        part.z = -sin(theta);
    } else {
        part.x = part.y = part.z = 0;
    }
}

//？求导？
void IncidentVector::calcDerivatives()
{
    calcCommonPart();
    derivatives.clear();
    derivatives.push_back(calcDu());
    derivatives.push_back(calcDv());
    derivatives.push_back(calcDf());
    std::vector<cv::Point3d> dak = calcDak();
    derivatives.insert(derivatives.end(), dak.begin(), dak.end());
}