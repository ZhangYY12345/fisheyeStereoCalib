//
//  incidentVector.h
//  Calibration
//
//  Created by Ryohei Suda on 2014/03/31.
//  Copyright (c) 2014 Ryohei Suda. All rights reserved.
//

#ifndef Calibration_IncidentVector_h
#define Calibration_IncidentVector_h

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#define PROJECTION_NUM 4

//������
class IncidentVector
{
protected:
    static double f; // Focal length (pixel unit)����
    static double f0; // Scale constant
    static std::vector<double> a; // Distortion parameters (odd)	�������ϵ����
    static std::vector<double> b; // Distortion parameters (even)	��ż����ϵ��
    static cv::Point2d center; // Optical center (u_0, v_0):��ѧ������ͼ����������ϵ�е����꣺ͼ������
    static cv::Size2i img_size; // Image size
    double theta;
    double r;
    static std::string projection_name[PROJECTION_NUM];
    static int projection; //Projection Model������ģ�ͣ���ѡ��ͶӰģ�͵ı�ʶ
    cv::Point3d part;	//���������theta�󵼽��

    void calcCommonPart(); // Calculate common part of derivatives
    virtual cv::Point3d calcDu() = 0;	//���ݲ�ͬ�ĳ���ģ�ͣ��ֱ�ʵ�ֺ���
    virtual cv::Point3d calcDv() = 0;
    virtual cv::Point3d calcDf() = 0;
    virtual std::vector<cv::Point3d> calcDak() = 0;
    
public:
    cv::Point3d m;
    cv::Point2d point;		//ͼ����������
    std::vector<cv::Point3d> derivatives;
    static int nparam; // Number of parameters (u0, v0, f, a1, a2, ...),����ڲ�
    
    IncidentVector(cv::Point2d p);
    
    // Setter and getter
    static void setParameters(double f, double f0, std::vector<double> a, cv::Size2i img_size, cv::Point2d center);
    static void setF(double f){ IncidentVector::f = f; }
    static double getF() { return IncidentVector::f; }
    static void setF0(double f0) { IncidentVector::f0 = f0; }
    static double getF0() { return IncidentVector::f0; }
    static void setA(std::vector<double> a) {
        IncidentVector::a = a;
        IncidentVector::nparam = 3 + (int)a.size();
    }
	//��ʼ������������ڲ�+�������
    static void initA(int a_size) {
        std::vector<double> a(a_size, 0);
        IncidentVector::a = a;
        IncidentVector::nparam = 3 + a_size;
    }
    static std::vector<double> getA() { return IncidentVector::a; }
    static void setB(std::vector<double> b) {
        IncidentVector::b = b;
        IncidentVector::nparam = 3 + (int)a.size() + (int)b.size();
    }
    static void initB(int b_size) {
        std::vector<double> b(b_size, 0);
        IncidentVector::b = b;
        IncidentVector::nparam = 3 + (int)a.size() + (int)b.size();
    }
    static std::vector<double> getB() { return IncidentVector::b; }
    static void setImgSize(cv::Size2i img_size) { IncidentVector::img_size = img_size; }
    static cv::Size2i getImgSize() { return IncidentVector::img_size; }
    static void setCenter(cv::Point2d c) { IncidentVector::center = c; }
    static cv::Point2d getCenter() { return IncidentVector::center; }
    static int A(int i) { return 3 + i; }
    static void setProjection(std::string projection);
    static int getProjection() { return projection; }
    static std::string getProjectionName() { return projection_name[projection]; }
    double getTheta() {
        return theta;
    }
    
    void calcDerivatives();
    void calcM();
    virtual double aoi(double r) = 0; // Calculate theta
};



#endif
