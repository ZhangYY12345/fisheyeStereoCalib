#include "fisheyeExpand.h"
#include "corrector.h"
#include "findCircleParameter.h"


void fisheyeExpand(cv::Mat src, cv::Mat& dst)
{
	bool isDispCorrectRet = false;

	correctParameters params;
	corrector adjuster;

	if (findCircleParameter::init(src))
	{
#pragma region У�������趨��
		params.imgOrg = src;
		//����ͼ�����Ч�����ֵ��ǰ�ڻ�ȡ��ֵ�����ͬһ����Ƕ�ֵ��//left camera
		params.center = cv::Point2i(1275, 708);
		params.radius = 1183;

		params.w_longtitude = PI / 2;
		params.w_latitude = PI / 2;
		params.distMap = LATITUDE_LONGTITUDE;
		params.theta_left = 0;
		params.phi_up = 0;
		params.camerFieldAngle = findCircleParameter::FOV;
		params.camProjMode = EQUIDISTANCE;
		params.typeOfCorrect = Reverse;
#pragma endregion			

#pragma region ͼ��У����
		corrector::correctMethod method = corrector::correctMethod::PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_HALF_PI;

		dst = adjuster.correctImage(params, method, isDispCorrectRet);
	}
}

void fisheyeExpand(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, corrector::correctMethod method_)
{
	bool isDispCorrectRet = false;

	correctParameters params;
	corrector adjuster;

	if (findCircleParameter::init(src))
	{
#pragma region У�������趨��
		params.imgOrg = src;
		//����ͼ�����Ч�����ֵ��ǰ�ڻ�ȡ��ֵ�����ͬһ����Ƕ�ֵ��//left camera
		params.center = cv::Point2i(1275, 708);
		params.radius = 1183;

		params.w_longtitude = PI / 2;
		params.w_latitude = PI / 2;
		params.distMap = LATITUDE_LONGTITUDE;
		params.theta_left = 0;
		params.phi_up = 0;
		params.camerFieldAngle = findCircleParameter::FOV;
		params.camProjMode = fisheyeCamModel;
		params.typeOfCorrect = Reverse;
#pragma endregion			

#pragma region ͼ��У����
		//corrector::correctMethod method = corrector::correctMethod::PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_HALF_PI;

		dst = adjuster.correctImage(params, method_, isDispCorrectRet);
	}
}

void fisheyeExpandTest(cv::Mat src, cv::Mat& dst, camMode fisheyeCamModel, std::string resPathPre)
{
	bool isDispCorrectRet = false;

	correctParameters params;
	corrector adjuster;

	if (findCircleParameter::init(src))
	{
#pragma region У�������趨��
		params.imgOrg = src;
		//����ͼ�����Ч�����ֵ��ǰ�ڻ�ȡ��ֵ�����ͬһ����Ƕ�ֵ��//left camera
		params.center = cv::Point2i(1275, 708);
		params.radius = 1183;

		params.w_longtitude = PI / 2;
		params.w_latitude = PI / 2;
		params.distMap = LATITUDE_LONGTITUDE;
		params.theta_left = 0;
		params.phi_up = 0;
		params.camerFieldAngle = findCircleParameter::FOV;
		params.camProjMode = fisheyeCamModel;
		params.typeOfCorrect = Reverse;
#pragma endregion			

#pragma region ͼ��У����
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

void fisheyeExpandApply(std::string imgPath)
{
	cv::String filePath = imgPath + "\\*L.jpg";
	std::vector<cv::String> fileNames;
	cv::glob(filePath, fileNames, false);
	cv::Mat image, imageRes;

	for (int i = 0; i < fileNames.size(); i++)
	{
		image = cv::imread(fileNames[i]);
		fisheyeExpand(image, imageRes);

		std::string resName = fileNames[i].substr(0, fileNames[i].length() - 4) + "_correct.jpg";
		cv::imwrite(resName, imageRes);
	}
}
