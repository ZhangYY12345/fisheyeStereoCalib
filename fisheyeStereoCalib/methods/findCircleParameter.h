#pragma once
#include "parametersStereo.h"

class findCircleParameter
{
private:
	findCircleParameter();
	~findCircleParameter();

	//为确定鱼眼圆形有效区域参数而设置的回调函数，供OpenCV使用
	static void On_N_trackbar(int N_slider_value, void* params);
	static void On_threshold_trackbar(int thresholdValue_slider_value, void* params);
	static void revisedScanLineMethod(cv::Mat imgOrg, cv::Point2i& center, int& radius, int threshold, int N);
	static bool CircleFitByKasa(std::vector<cv::Point> validPoints, cv::Point& center, int&	 radius);
	static void onMouse(int event, int x, int y, int, void* args);
	static void findPoints(cv::Point2i center, int radius, std::vector<cv::Point> &points, camMode projMode = ORTHOGONAL);

public:
	static bool init(cv::Mat img);
	static void findCircle();
	static void checkVarify();
	static bool getCircleParatemer(cv::Point2i& center, int& radius);
private:
	/*提取圆参数*/
	static std::string win_name;
	static std::string N_trackbar_name;
	static std::string thresholdValue_trackbar_name;
	static int N_slider_value;
	static int N_max_value;
	static int thresholdValue_slider_value;
	static int thresholdValue_max_value;
	static cv::Point2i center;
	static int radius;
	static cv::Mat image;

	/*验证圆*/
	static std::vector<std::vector<cv::Point>> lines;
	static std::vector<cv::Point> points;
	static std::string check_win_name;

	static int width_disp_img;
	static int height_disp_img;
public:
	static const double FOV;
};

