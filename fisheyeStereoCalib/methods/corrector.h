#pragma once
#include "parametersStereo.h"

typedef struct correctParameters
{
	cv::Mat imgOrg;
	cv::Point2i center;
	int radius; 
	double w_longtitude; 
	double w_latitude; 
	distMapMode distMap ; 
	double theta_left ; 
	double phi_up ; 
	double camerFieldAngle ; 
	camMode camProjMode;
	CorrectType typeOfCorrect;
}correctParams;

class corrector
{
public:
	enum correctMethod
	{
		LONG_LAT_MAP_REVERSE_FORWARD,
		PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL,
		PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_HALF_PI,
		PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_VARIABLE,
		PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_FORWORD_W_VARIABLE
	};

public:
	corrector(){}
	~corrector(){};
	static void dispHeaveAndEarthCorrectImage(cv::Mat sourceImage);
	cv::Mat correctImage(correctParameters params, correctMethod method, bool isDispRet);
private:
	static cv::Mat heavenAndEarthCorrect(cv::Mat imgOrg, cv::Point center, int radius, double startRadian = 0, CorrectType type = Reverse);
	
private:

#pragma region 鱼眼图像校正部分成员函数
	cv::Mat latitudeCorrection(cv::Mat imgOrg, cv::Point2i center, int radius, double camerFieldAngle = PI, CorrectType type = Reverse);
	cv::Mat latitudeCorrection2(cv::Mat imgOrg, cv::Point2i center, int radius, distMapMode distMap = LATITUDE_LONGTITUDE, double camerFieldAngle = PI, camMode camProjMode = EQUIDISTANCE);
	cv::Mat latitudeCorrection3(cv::Mat imgOrg, cv::Point2i center, int radius, distMapMode distMap = LATITUDE_LONGTITUDE, double theta_left = 0, double phi_up = 0, double camerFieldAngle = PI, camMode camProjMode = EQUIDISTANCE);
	 static double func(double l, double phi);
	 static double getPhi(double l);

	 cv::Mat latitudeCorrection4(cv::Mat imgOrg, cv::Point2i center, int radius, double w_longtitude, double w_latitude, distMapMode distMap = LATITUDE_LONGTITUDE, double theta_left = 0, double phi_up = 0, double camerFieldAngle = PI, camMode camProjMode = EQUIDISTANCE);

	 static double func1(double l, double phi, double w);
	 static double getPhi1(double l, double w);
	 static double auxFunc(double w, double phi);

	 cv::Mat latitudeCorrection5(cv::Mat imgOrg, cv::Point2i center, int radius, double w_longtitude, double w_latitude, distMapMode distMap = LATITUDE_LONGTITUDE, double theta_left = 0, double phi_up = 0, double camerFieldAngle = PI, camMode camProjMode = EQUIDISTANCE);
#pragma endregion
private:
	static int counter;
};
