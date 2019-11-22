#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "../parametersStereo.h"

//distort 
namespace my_cv
{
	namespace internal {
		struct CV_EXPORTS IntrinsicParams
		{
			cv::Vec2d f;
			cv::Vec2d c;
			cv::Vec4d k;
			double alpha;
			std::vector<uchar> isEstimate;

			IntrinsicParams();
			IntrinsicParams(cv::Vec2d f, cv::Vec2d c, cv::Vec4d k, double alpha = 0);
			IntrinsicParams operator+(const cv::Mat& a);
			IntrinsicParams& operator =(const cv::Mat& a);
			void Init(const cv::Vec2d& f, const cv::Vec2d& c, const cv::Vec4d& k = cv::Vec4d(0, 0, 0, 0), const double& alpha = 0);
		};
		void projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints,
			cv::InputArray _rvec, cv::InputArray _tvec,
			const IntrinsicParams& param, cv::OutputArray jacobian);

		void ComputeExtrinsicRefine(const cv::Mat& imagePoints, const cv::Mat& objectPoints, cv::Mat& rvec,
		                            cv::Mat&  tvec, cv::Mat& J, const int MaxIter,
			const IntrinsicParams& param, const double thresh_cond);
		CV_EXPORTS cv::Mat ComputeHomography(cv::Mat m, cv::Mat M);

		CV_EXPORTS cv::Mat NormalizePixels(const cv::Mat& imagePoints, const IntrinsicParams& param);

		void InitExtrinsics(const cv::Mat& _imagePoints, const cv::Mat& _objectPoints, const IntrinsicParams& param, cv::Mat& omckk, cv::Mat& Tckk);

		void CalibrateExtrinsics(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
			const IntrinsicParams& param, const int check_cond,
			const double thresh_cond, cv::InputOutputArray omc, cv::InputOutputArray Tc);

		void ComputeJacobians(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
			const IntrinsicParams& param, cv::InputArray omc, cv::InputArray Tc,
			const int& check_cond, const double& thresh_cond, cv::Mat& JJ2_inv, cv::Mat& ex3);

		CV_EXPORTS void  EstimateUncertainties(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
			const IntrinsicParams& params, cv::InputArray omc, cv::InputArray Tc,
			IntrinsicParams& errors, cv::Vec2d& std_err, double thresh_cond, int check_cond, double& rms);

		void dAB(cv::InputArray A, cv::InputArray B, cv::OutputArray dABdA, cv::OutputArray dABdB);

		void JRodriguesMatlab(const cv::Mat& src, cv::Mat& dst);

		void compose_motion(cv::InputArray _om1, cv::InputArray _T1, cv::InputArray _om2, cv::InputArray _T2,
		                    cv::Mat& om3, cv::Mat& T3, cv::Mat& dom3dom1, cv::Mat& dom3dT1, cv::Mat& dom3dom2,
			cv::Mat& dom3dT2, cv::Mat& dT3dom1, cv::Mat& dT3dT1, cv::Mat& dT3dom2, cv::Mat& dT3dT2);

		double median(const cv::Mat& row);

		cv::Vec3d median3d(cv::InputArray m);

	}

	namespace fisheye
	{
		CV_EXPORTS void projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints, const cv::Affine3d& affine,
		                              cv::InputArray K, cv::InputArray D, double alpha = 0, cv::OutputArray jacobian = cv::noArray(), camMode mode = STEREOGRAPHIC);
		CV_EXPORTS_W void projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints, cv::InputArray rvec, cv::InputArray tvec,
		                                cv::InputArray K, cv::InputArray D, double alpha = 0, cv::OutputArray jacobian = cv::noArray(), camMode mode = STEREOGRAPHIC);

		CV_EXPORTS_W void distortPoints(cv::InputArray undistorted, cv::OutputArray distorted, cv::InputArray K, cv::InputArray D, double alpha = 0, camMode mode = STEREOGRAPHIC);

		CV_EXPORTS_W void undistortPoints(cv::InputArray distorted, cv::OutputArray undistorted,
		                                  cv::InputArray K, cv::InputArray D, cv::InputArray R = cv::noArray(), cv::InputArray P = cv::noArray(), camMode mode = STEREOGRAPHIC);

		CV_EXPORTS_W void initUndistortRectifyMap(cv::InputArray K, cv::InputArray D, cv::InputArray R, cv::InputArray P,
			const cv::Size& size, int m1type, cv::OutputArray map1, cv::OutputArray map2, camMode mode = STEREOGRAPHIC);

		CV_EXPORTS_W void undistortImage(cv::InputArray distorted, cv::OutputArray undistorted,
		                                 cv::InputArray K, cv::InputArray D, cv::InputArray Knew = cv::noArray(), const cv::Size& new_size = cv::Size());

		CV_EXPORTS_W void estimateNewCameraMatrixForUndistortRectify(cv::InputArray K, cv::InputArray D, const cv::Size &image_size, cv::InputArray R,
		                                                             cv::OutputArray P, double balance = 0.0, const cv::Size& new_size = cv::Size(), double fov_scale = 1.0);

		CV_EXPORTS_W double calibrate(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints, const cv::Size& image_size,
		                              cv::InputOutputArray K, cv::InputOutputArray D, cv::OutputArrayOfArrays rvecs, cv::OutputArrayOfArrays tvecs, int flags = 0,
		                              cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON));

		CV_EXPORTS_W void stereoRectify(cv::InputArray K1, cv::InputArray D1, cv::InputArray K2, cv::InputArray D2, const cv::Size &imageSize, cv::InputArray R, cv::InputArray tvec,
		                                cv::OutputArray R1, cv::OutputArray R2, cv::OutputArray P1, cv::OutputArray P2, cv::OutputArray Q, int flags, const cv::Size &newImageSize = cv::Size(),
			double balance = 0.0, double fov_scale = 1.0);

		CV_EXPORTS_W double stereoCalibrate(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints1, cv::InputArrayOfArrays imagePoints2,
			cv::InputOutputArray K1, cv::InputOutputArray D1, cv::InputOutputArray K2, cv::InputOutputArray D2, cv::Size imageSize,
			cv::OutputArray R, cv::OutputArray T, int flags = cv::fisheye::CALIB_FIX_INTRINSIC,
			cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON));
	}
}