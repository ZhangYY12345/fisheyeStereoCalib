#include "fisheyeExpand.h"
#include "corrector.h"
#include "findCircleParameter.h"

// for distorted fisheye images
// for left camera:
// center = cv::Point(1283, 724)
// radius = 1199
// for right camera
// center = cv::Point(1283, 722)
// radius = 1187


// for undistorted fisheye images
// for ideal equidistance model and ideal equisolidAngle model
// center = cv::Point(1560, 940)
// radius = 1286

// for ideal stereographic model
// center = cv::Point(2560, 1440)
// radius = 2270

void fisheyeExpand(cv::Mat src, cv::Mat& dst, cv::Point2i center, int radius)
{
	bool isDispCorrectRet = false;

	correctParameters params;
	corrector adjuster;

	if (findCircleParameter::init(src))
	{
#pragma region 校正参数设定区
		params.imgOrg = src;
		//鱼眼图像的有效区域的值是前期获取的值（针对同一相机是定值）
		params.center = center;
		params.radius = radius;

		params.w_longtitude = PI / 2;
		params.w_latitude = PI / 2;
		params.distMap = LATITUDE_LONGTITUDE;
		params.theta_left = 0;
		params.phi_up = 0;
		params.camerFieldAngle = findCircleParameter::FOV;
		params.camProjMode = EQUIDISTANCE;
		params.typeOfCorrect = Reverse;
#pragma endregion			

#pragma region 图像校正区
		corrector::correctMethod method = corrector::correctMethod::PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_HALF_PI;

		dst = adjuster.correctImage(params, method, isDispCorrectRet);
	}
}

void fisheyeExpand(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, corrector::correctMethod method_, cv::Point2i center, int radius)
{
	bool isDispCorrectRet = false;

	correctParameters params;
	corrector adjuster;

	if (findCircleParameter::init(src))
	{
#pragma region 校正参数设定区
		params.imgOrg = src;

		//鱼眼图像的有效区域的值是前期获取的值（针对同一相机是定值）
		params.center = center;
		params.radius = radius;

		params.w_longtitude = PI / 2;
		params.w_latitude = PI / 2;
		params.distMap = LATITUDE_LONGTITUDE;
		params.theta_left = 0;
		params.phi_up = 0;
		params.camerFieldAngle = findCircleParameter::FOV;
		params.camProjMode = fisheyeCamModel;
		params.typeOfCorrect = Reverse;
#pragma endregion			

#pragma region 图像校正区
		//corrector::correctMethod method = corrector::correctMethod::PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_HALF_PI;

		dst = adjuster.correctImage(params, method_, isDispCorrectRet);
	}
}

void fisheyeExpandTest(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, std::string resPathPre, cv::Point2i center, int radius)
{
	bool isDispCorrectRet = false;

	correctParameters params;
	corrector adjuster;

	if (findCircleParameter::init(src))
	{
#pragma region 校正参数设定区
		params.imgOrg = src;
		//鱼眼图像的有效区域的值是前期获取的值（针对同一相机是定值）
		params.center = center;
		params.radius = radius;

		params.w_longtitude = PI / 2;
		params.w_latitude = PI / 2;
		params.distMap = LATITUDE_LONGTITUDE;
		params.theta_left = 0;
		params.phi_up = 0;
		params.camerFieldAngle = findCircleParameter::FOV;
		params.camProjMode = fisheyeCamModel;
		params.typeOfCorrect = Reverse;
#pragma endregion			

#pragma region 图像校正区
		//params.typeOfCorrect = Forward;
		corrector::correctMethod method = corrector::correctMethod::LONG_LAT_MAP_REVERSE_FORWARD;
		dst = adjuster.correctImage(params, method, isDispCorrectRet);
		cv::imwrite(resPathPre + "_LLMRF.jpg", dst);

		method = corrector::correctMethod::PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_FORWORD_W_VARIABLE;
		dst = adjuster.correctImage(params, method, isDispCorrectRet);
		cv::imwrite(resPathPre + "_PLLMCLMFWV.jpg", dst);


		
		method = corrector::correctMethod::PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL;
		dst = adjuster.correctImage(params, method, isDispCorrectRet);
		cv::imwrite(resPathPre + "_PLLMCLM.jpg", dst);

		method = corrector::correctMethod::PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_HALF_PI;
		dst = adjuster.correctImage(params, method, isDispCorrectRet);
		cv::imwrite(resPathPre + "_PLLMCLMRWHP.jpg", dst);

		method = corrector::correctMethod::PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_VARIABLE;
		dst = adjuster.correctImage(params, method, isDispCorrectRet);
		cv::imwrite(resPathPre + "_PLLMCLMRWV.jpg", dst);

	}

}

void fisheyeExpandApply(std::string imgPath, cv::Point2i center, int radius)
{
	cv::String filePath = imgPath + "\\*L.jpg";
	std::vector<cv::String> fileNames;
	cv::glob(filePath, fileNames, false);
	cv::Mat image, imageRes;

	for (int i = 0; i < fileNames.size(); i++)
	{
		image = cv::imread(fileNames[i]);
		fisheyeExpand(image, imageRes, center, radius);

		std::string resName = fileNames[i].substr(0, fileNames[i].length() - 4) + "_correct.jpg";
		cv::imwrite(resName, imageRes);
	}
}
