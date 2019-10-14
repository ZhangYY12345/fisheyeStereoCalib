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
