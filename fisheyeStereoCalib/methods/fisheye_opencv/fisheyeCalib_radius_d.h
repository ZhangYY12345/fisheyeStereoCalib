#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/core/affine.hpp>
#include "../parametersStereo.h"

namespace my_cv
{
	// r_d = r(1 + k[0] * r^2 + k[1] * r^4 + k[2] * r^6 + k[3] * r^8)//目前测试结果显示，只适用于 等距投影模型
	// r_d = k[0] + k[1]*r + k[2]*r2 + k[3]*r3 + k[4]*r4 + k[5]*r5
	namespace fisheye_r_d
	{
		CV_EXPORTS void projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints, const cv::Affine3d& affine,
			cv::InputArray K, cv::InputArray D, cv::InputArray DE, double alpha = 0, cv::OutputArray jacobian = cv::noArray(), camMode mode = STEREOGRAPHIC);
		CV_EXPORTS_W void projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints, cv::InputArray rvec, cv::InputArray tvec,
			cv::InputArray K, cv::InputArray D, cv::InputArray DE, double alpha = 0, cv::OutputArray jacobian = cv::noArray(), camMode mode = STEREOGRAPHIC);

		CV_EXPORTS_W void distortPoints(cv::InputArray undistorted, cv::OutputArray distorted, cv::InputArray K, cv::InputArray D, cv::InputArray DE, double alpha = 0, camMode mode = STEREOGRAPHIC);

		CV_EXPORTS_W void undistortPoints(cv::InputArray distorted, cv::OutputArray undistorted,
			cv::InputArray K, cv::InputArray D, cv::InputArray DE, cv::InputArray R = cv::noArray(), cv::InputArray P = cv::noArray(), camMode mode = STEREOGRAPHIC);
		CV_EXPORTS_W void undistortPoints_H(cv::InputArray distorted, cv::OutputArray undistorted,
			cv::InputArray K, cv::InputArray D, cv::InputArray DE, camMode mode = STEREOGRAPHIC);

		CV_EXPORTS_W void initUndistortRectifyMap(cv::InputArray K, cv::InputArray D, cv::InputArray DE, cv::InputArray R, cv::InputArray P,
			const cv::Size& size, int m1type, cv::OutputArray map1, cv::OutputArray map2, camMode mode = STEREOGRAPHIC);

		CV_EXPORTS_W void undistortImage(cv::InputArray distorted, cv::OutputArray undistorted,
			cv::InputArray K, cv::InputArray D, cv::InputArray DE, cv::InputArray Knew = cv::noArray(), const cv::Size& new_size = cv::Size());

		CV_EXPORTS_W void estimateNewCameraMatrixForUndistortRectify(cv::InputArray K, cv::InputArray D, cv::InputArray DE, const cv::Size &image_size, cv::InputArray R,
			cv::OutputArray P, double balance = 0.0, const cv::Size& new_size = cv::Size(), double fov_scale = 1.0);

		CV_EXPORTS_W double calibrate(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints, const cv::Size& image_size,
			cv::InputOutputArray K, cv::InputOutputArray D, cv::InputOutputArray DE, cv::OutputArrayOfArrays rvecs, cv::OutputArrayOfArrays tvecs, int flags = 0,
			cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON));

		CV_EXPORTS_W void stereoRectify(cv::InputArray K1, cv::InputArray D1, cv::InputArray DE1, cv::InputArray K2, cv::InputArray D2, cv::InputArray DE2, const cv::Size &imageSize, cv::InputArray R, cv::InputArray tvec,
			cv::OutputArray R1, cv::OutputArray R2, cv::OutputArray P1, cv::OutputArray P2, cv::OutputArray Q, int flags, const cv::Size &newImageSize = cv::Size(),
			double balance = 0.0, double fov_scale = 1.0);

		CV_EXPORTS_W double stereoCalibrate(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints1, cv::InputArrayOfArrays imagePoints2,
			cv::InputOutputArray K1, cv::InputOutputArray D1, cv::InputOutputArray DE1, cv::InputOutputArray K2, cv::InputOutputArray D2, cv::InputOutputArray DE2, cv::Size imageSize,
			cv::OutputArray R, cv::OutputArray T, int flags = cv::fisheye::CALIB_FIX_INTRINSIC,
			cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON));
	}
}