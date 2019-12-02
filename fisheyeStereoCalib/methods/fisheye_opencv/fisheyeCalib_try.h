#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "../parametersStereo.h"

//distort 
namespace my_cv
{
	namespace
	{
		struct JacobianRow
		{
			cv::Vec2d df, dc;
			cv::Vec6d dk;
			cv::Vec3d dom, dT;
			double dalpha;
		};

		void subMatrix(const cv::Mat& src, cv::Mat& dst, const std::vector<uchar>& cols, const std::vector<uchar>& rows)
		{
			CV_Assert(src.channels() == 1);

			int nonzeros_cols = cv::countNonZero(cols);
			cv::Mat tmp(src.rows, nonzeros_cols, CV_64F);

			for (int i = 0, j = 0; i < (int)cols.size(); i++)
			{
				if (cols[i])
				{
					src.col(i).copyTo(tmp.col(j++));
				}
			}

			int nonzeros_rows = cv::countNonZero(rows);
			dst.create(nonzeros_rows, nonzeros_cols, CV_64F);
			for (int i = 0, j = 0; i < (int)rows.size(); i++)
			{
				if (rows[i])
				{
					tmp.row(i).copyTo(dst.row(j++));
				}
			}
		}
	}

	namespace internal {
		struct CV_EXPORTS IntrinsicParams
		{
			cv::Vec2d f;
			cv::Vec2d c;
			cv::Vec6d k;
			double alpha;
			std::vector<uchar> isEstimate;

			IntrinsicParams();
			IntrinsicParams(cv::Vec2d f, cv::Vec2d c, cv::Vec6d k, double alpha = 0);
			IntrinsicParams operator+(const cv::Mat& a);
			IntrinsicParams& operator =(const cv::Mat& a);
			void Init(const cv::Vec2d& f, const cv::Vec2d& c, const cv::Vec6d& k = cv::Vec6d(0, 0, 0, 0, 0, 0), const double& alpha = 0);
		};
		void projectPoints(cv::InputOutputArray objectPoints, cv::InputOutputArray imagePoints,
			cv::InputArray _rvec, cv::InputArray _tvec,
			const IntrinsicParams& param, cv::OutputArray jacobian, DISTORT_Mode_Fisheye distortMode);

		void ComputeExtrinsicRefine(const cv::Mat& imagePoints, const cv::Mat& objectPoints, cv::Mat& rvec,
		                            cv::Mat&  tvec, cv::Mat& J, const int MaxIter,
			const IntrinsicParams& param, const double thresh_cond, DISTORT_Mode_Fisheye distortMode);
		CV_EXPORTS cv::Mat ComputeHomography(cv::Mat m, cv::Mat M);

		CV_EXPORTS cv::Mat NormalizePixels(const cv::Mat& imagePoints, 
			const IntrinsicParams& param, DISTORT_Mode_Fisheye distortMode);

		void InitExtrinsics(const cv::Mat& _imagePoints, const cv::Mat& _objectPoints,
			const IntrinsicParams& param, cv::Mat& omckk, cv::Mat& Tckk, DISTORT_Mode_Fisheye distortMode);

		void CalibrateExtrinsics(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
			const IntrinsicParams& param, const int check_cond,
			const double thresh_cond, cv::InputOutputArray omc, cv::InputOutputArray Tc, 
			DISTORT_Mode_Fisheye distortMode);

		void ComputeJacobians(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
			const IntrinsicParams& param, cv::InputArray omc, cv::InputArray Tc,
			const int& check_cond, const double& thresh_cond, cv::Mat& JJ2_inv, cv::Mat& ex3,
			DISTORT_Mode_Fisheye distortMode);

		CV_EXPORTS void  EstimateUncertainties(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
			const IntrinsicParams& params, cv::InputArray omc, cv::InputArray Tc,
			IntrinsicParams& errors, cv::Vec2d& std_err, double thresh_cond, int check_cond, double& rms,
			DISTORT_Mode_Fisheye distortMode);
		CV_EXPORTS void  EstimateUncertainties_rd(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
			const IntrinsicParams& params, cv::InputArray omc, cv::InputArray Tc,
			IntrinsicParams& errors, cv::Vec3d& std_err, double thresh_cond, int check_cond, double& rms,
			DISTORT_Mode_Fisheye distortMode);

		void dAB(cv::InputArray A, cv::InputArray B, cv::OutputArray dABdA, cv::OutputArray dABdB);

		void JRodriguesMatlab(const cv::Mat& src, cv::Mat& dst);

		void compose_motion(cv::InputArray _om1, cv::InputArray _T1, cv::InputArray _om2, cv::InputArray _T2,
		                    cv::Mat& om3, cv::Mat& T3, cv::Mat& dom3dom1, cv::Mat& dom3dT1, cv::Mat& dom3dom2,
			cv::Mat& dom3dT2, cv::Mat& dT3dom1, cv::Mat& dT3dT1, cv::Mat& dT3dom2, cv::Mat& dT3dT2);

		double median(const cv::Mat& row);

		cv::Vec3d median3d(cv::InputArray m);

	}
}

double getR(double theta, camMode mode);
double getTheta(double r, camMode mode);
double get_drdtheta(double theta, camMode mode);
double get_dthetadr(double r, camMode mode);
