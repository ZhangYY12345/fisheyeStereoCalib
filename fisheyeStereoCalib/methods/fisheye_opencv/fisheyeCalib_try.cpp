#include "fisheyeCalib_try.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "fisheyeCalib_theta_d.h"
#include "fisheyeCalib_radius_d.h"
#include "fisheyeCalib_radius_rd.h"

camMode cur_fisheye_mode = STEREOGRAPHIC;

my_cv::internal::IntrinsicParams::IntrinsicParams() :
	f(cv::Vec2d::all(0)), c(cv::Vec2d::all(0)), k(cv::Vec6d::all(0)), alpha(0), isEstimate(11, 0)
{
}

my_cv::internal::IntrinsicParams::IntrinsicParams(cv::Vec2d _f, cv::Vec2d _c, cv::Vec6d _k, double _alpha) :
	f(_f), c(_c), k(_k), alpha(_alpha), isEstimate(11, 0)
{
}

my_cv::internal::IntrinsicParams my_cv::internal::IntrinsicParams::operator+(const cv::Mat& a)
{
	CV_Assert(a.type() == CV_64FC1);
	IntrinsicParams tmp;
	const double* ptr = a.ptr<double>();

	int j = 0;
	tmp.f[0] = this->f[0] + (isEstimate[0] ? ptr[j++] : 0);
	tmp.f[1] = this->f[1] + (isEstimate[1] ? ptr[j++] : 0);
	tmp.c[0] = this->c[0] + (isEstimate[2] ? ptr[j++] : 0);
	tmp.c[1] = this->c[1] + (isEstimate[3] ? ptr[j++] : 0);
	tmp.alpha = this->alpha + (isEstimate[4] ? ptr[j++] : 0);
	tmp.k[0] = this->k[0] + (isEstimate[5] ? ptr[j++] : 0);
	tmp.k[1] = this->k[1] + (isEstimate[6] ? ptr[j++] : 0);
	tmp.k[2] = this->k[2] + (isEstimate[7] ? ptr[j++] : 0);
	tmp.k[3] = this->k[3] + (isEstimate[8] ? ptr[j++] : 0);
	tmp.k[4] = this->k[4] + (isEstimate[9] ? ptr[j++] : 0);
	tmp.k[5] = this->k[5] + (isEstimate[10] ? ptr[j++] : 0);

	tmp.isEstimate = isEstimate;
	return tmp;
}

my_cv::internal::IntrinsicParams& my_cv::internal::IntrinsicParams::operator =(const cv::Mat& a)
{
	CV_Assert(a.type() == CV_64FC1);
	const double* ptr = a.ptr<double>();

	int j = 0;

	this->f[0] = isEstimate[0] ? ptr[j++] : 0;
	this->f[1] = isEstimate[1] ? ptr[j++] : 0;
	this->c[0] = isEstimate[2] ? ptr[j++] : 0;
	this->c[1] = isEstimate[3] ? ptr[j++] : 0;
	this->alpha = isEstimate[4] ? ptr[j++] : 0;
	this->k[0] = isEstimate[5] ? ptr[j++] : 0;
	this->k[1] = isEstimate[6] ? ptr[j++] : 0;
	this->k[2] = isEstimate[7] ? ptr[j++] : 0;
	this->k[3] = isEstimate[8] ? ptr[j++] : 0;
	this->k[4] = isEstimate[9] ? ptr[j++] : 0;
	this->k[5] = isEstimate[10] ? ptr[j++] : 0;

	return *this;
}

void my_cv::internal::IntrinsicParams::Init(const cv::Vec2d& _f, const cv::Vec2d& _c, const cv::Vec6d& _k, const double& _alpha)
{
	this->c = _c;
	this->f = _f;
	this->k = _k;
	this->alpha = _alpha;
}

void my_cv::internal::projectPoints(cv::InputOutputArray objectPoints, cv::InputOutputArray imagePoints,
	cv::InputArray _rvec, cv::InputArray _tvec,
	const IntrinsicParams& param, cv::OutputArray jacobian, DISTORT_Mode_Fisheye distortMode)
{
	if (distortMode == ORIGIN_FISHEYE_CALIB || distortMode == THETA_D_FISHEYE_CALIB
		|| distortMode == RADIUS_D_FISHEYE_CALIB)
	{
		CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
	}
	else if(distortMode == RADIUS_RD_FISHEYE_CALIB)
	{
		CV_Assert(!imagePoints.empty() && (imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2));
	}

	cv::Matx33d K(param.f[0], param.f[0] * param.alpha, param.c[0],
		0, param.f[1], param.c[1],
		0, 0, 1);
	switch(distortMode)
	{
	case ORIGIN_FISHEYE_CALIB:
		cv::fisheye::projectPoints(objectPoints, imagePoints, _rvec, _tvec, K, param.k, param.alpha, jacobian);//////
		break;
	case THETA_D_FISHEYE_CALIB:
		my_cv::fisheye::projectPoints(objectPoints, imagePoints, _rvec, _tvec, K, param.k, param.alpha, jacobian, cur_fisheye_mode);//////
		break;
	case RADIUS_D_FISHEYE_CALIB:
		my_cv::fisheye_r_d::projectPoints(objectPoints, imagePoints, _rvec, _tvec, K, param.k, param.alpha, jacobian, cur_fisheye_mode);//////
		break;
	case RADIUS_RD_FISHEYE_CALIB:
		my_cv::fisheye_r_rd::projectPoints(imagePoints, objectPoints, _rvec, _tvec, K, param.k, param.alpha, jacobian, cur_fisheye_mode);//////
		break;
	}
}


void my_cv::internal::ComputeExtrinsicRefine(const cv::Mat& imagePoints, const cv::Mat& objectPoints, cv::Mat& rvec,
	cv::Mat&  tvec, cv::Mat& J, const int MaxIter,
	const IntrinsicParams& param, const double thresh_cond, DISTORT_Mode_Fisheye distortMode)
{
	CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
	CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);
	CV_Assert(rvec.total() > 2 && tvec.total() > 2);
	cv::Vec6d extrinsics(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
		tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
	double change = 1;
	int iter = 0;

	if (distortMode != RADIUS_RD_FISHEYE_CALIB)
	{
		while (change > 1e-10 && iter < MaxIter)
		{
			std::vector<cv::Point2d> x;
			cv::Mat jacobians;
			projectPoints(objectPoints, x, rvec, tvec, param, jacobians, distortMode);

			cv::Mat ex = imagePoints - cv::Mat(x).t();
			ex = ex.reshape(1, 2);

			J = jacobians.colRange(10, 16).clone();

			cv::SVD svd(J, cv::SVD::NO_UV);
			double condJJ = svd.w.at<double>(0) / svd.w.at<double>(5);

			if (condJJ > thresh_cond)
				change = 0;
			else
			{
				cv::Vec6d param_innov;
				solve(J, ex.reshape(1, (int)ex.total()), param_innov, cv::DECOMP_SVD + cv::DECOMP_NORMAL);

				cv::Vec6d param_up = extrinsics + param_innov;
				change = cv::norm(param_innov) / cv::norm(param_up);
				extrinsics = param_up;
				iter = iter + 1;

				rvec = cv::Mat(cv::Vec3d(extrinsics.val));
				tvec = cv::Mat(cv::Vec3d(extrinsics.val + 3));
			}
		}
	}
	else
	{
		while (change > 1e-10 && iter < MaxIter)
		{
			std::vector<cv::Point3d> x;
			cv::Mat jacobians;
			projectPoints(x, imagePoints, rvec, tvec, param, jacobians, distortMode);

			cv::Mat ex = objectPoints - cv::Mat(x).t();
			ex = ex.reshape(1, 3);

			J = jacobians.colRange(10, 16).clone();

			cv::SVD svd(J, cv::SVD::NO_UV);
			double condJJ = svd.w.at<double>(0) / svd.w.at<double>(5);

			if (condJJ > thresh_cond)
				change = 0;
			else
			{
				cv::Vec6d param_innov;
				solve(J, ex.reshape(1, (int)ex.total()), param_innov, cv::DECOMP_SVD + cv::DECOMP_NORMAL);

				cv::Vec6d param_up = extrinsics + param_innov;
				change = cv::norm(param_innov) / cv::norm(param_up);
				extrinsics = param_up;
				iter = iter + 1;

				rvec = cv::Mat(cv::Vec3d(extrinsics.val));
				tvec = cv::Mat(cv::Vec3d(extrinsics.val + 3));
			}
		}
	}
}

cv::Mat my_cv::internal::ComputeHomography(cv::Mat m, cv::Mat M)
{
	int Np = m.cols;

	if (m.rows < 3)
	{
		vconcat(m, cv::Mat::ones(1, Np, CV_64FC1), m);
	}
	if (M.rows < 3)
	{
		vconcat(M, cv::Mat::ones(1, Np, CV_64FC1), M);
	}

	divide(m, cv::Mat::ones(3, 1, CV_64FC1) * m.row(2), m);
	divide(M, cv::Mat::ones(3, 1, CV_64FC1) * M.row(2), M);

	cv::Mat ax = m.row(0).clone();
	cv::Mat ay = m.row(1).clone();

	double mxx = mean(ax)[0];
	double myy = mean(ay)[0];

	ax = ax - mxx;
	ay = ay - myy;

	double scxx = mean(abs(ax))[0];
	double scyy = mean(abs(ay))[0];

	cv::Mat Hnorm(cv::Matx33d(1 / scxx, 0.0, -mxx / scxx,
		0.0, 1 / scyy, -myy / scyy,
		0.0, 0.0, 1.0));

	cv::Mat inv_Hnorm(cv::Matx33d(scxx, 0, mxx,
		0, scyy, myy,
		0, 0, 1));
	cv::Mat mn = Hnorm * m;

	cv::Mat L = cv::Mat::zeros(2 * Np, 9, CV_64FC1);

	for (int i = 0; i < Np; ++i)
	{ 
		for (int j = 0; j < 3; j++)
		{
			L.at<double>(2 * i, j) = M.at<double>(j, i);
			L.at<double>(2 * i + 1, j + 3) = M.at<double>(j, i);
			L.at<double>(2 * i, j + 6) = -mn.at<double>(0, i) * M.at<double>(j, i);
			L.at<double>(2 * i + 1, j + 6) = -mn.at<double>(1, i) * M.at<double>(j, i);
		}
	}

	if (Np > 4) L = L.t() * L;
	cv::SVD svd(L);
	cv::Mat hh = svd.vt.row(8) / svd.vt.row(8).at<double>(8);
	cv::Mat Hrem = hh.reshape(1, 3);
	cv::Mat H = inv_Hnorm * Hrem;

	if (Np > 4)
	{
		cv::Mat hhv = H.reshape(1, 9)(cv::Rect(0, 0, 1, 8)).clone();
		for (int iter = 0; iter < 10; iter++)
		{
			cv::Mat mrep = H * M;
			cv::Mat J = cv::Mat::zeros(2 * Np, 8, CV_64FC1);
			cv::Mat MMM;
			divide(M, cv::Mat::ones(3, 1, CV_64FC1) * mrep(cv::Rect(0, 2, mrep.cols, 1)), MMM);
			divide(mrep, cv::Mat::ones(3, 1, CV_64FC1) * mrep(cv::Rect(0, 2, mrep.cols, 1)), mrep);
			cv::Mat m_err = m(cv::Rect(0, 0, m.cols, 2)) - mrep(cv::Rect(0, 0, mrep.cols, 2));
			m_err = cv::Mat(m_err.t()).reshape(1, m_err.cols * m_err.rows);
			cv::Mat MMM2, MMM3;
			multiply(cv::Mat::ones(3, 1, CV_64FC1) * mrep(cv::Rect(0, 0, mrep.cols, 1)), MMM, MMM2);
			multiply(cv::Mat::ones(3, 1, CV_64FC1) * mrep(cv::Rect(0, 1, mrep.cols, 1)), MMM, MMM3);

			for (int i = 0; i < Np; ++i)
			{
				for (int j = 0; j < 3; ++j)
				{
					J.at<double>(2 * i, j) = -MMM.at<double>(j, i);
					J.at<double>(2 * i + 1, j + 3) = -MMM.at<double>(j, i);
				}

				for (int j = 0; j < 2; ++j)
				{
					J.at<double>(2 * i, j + 6) = MMM2.at<double>(j, i);
					J.at<double>(2 * i + 1, j + 6) = MMM3.at<double>(j, i);
				}
			}
			divide(M, cv::Mat::ones(3, 1, CV_64FC1) * mrep(cv::Rect(0, 2, mrep.cols, 1)), MMM);
			cv::Mat hh_innov = (J.t() * J).inv() * (J.t()) * m_err;
			cv::Mat hhv_up = hhv - hh_innov;
			cv::Mat tmp;
			vconcat(hhv_up, cv::Mat::ones(1, 1, CV_64FC1), tmp);
			cv::Mat H_up = tmp.reshape(1, 3);
			hhv = hhv_up;
			H = H_up;
		}
	}
	return H;
}

cv::Mat my_cv::internal::NormalizePixels(const cv::Mat& imagePoints, const IntrinsicParams& param, DISTORT_Mode_Fisheye distortMode)
{
	CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);

	cv::Mat distorted((int)imagePoints.total(), 1, CV_64FC2), undistorted;
	const cv::Vec2d* ptr = imagePoints.ptr<cv::Vec2d>();      //图像像素坐标
	cv::Vec2d* ptr_d = distorted.ptr<cv::Vec2d>();              //
	for (size_t i = 0; i < imagePoints.total(); ++i)
	{
		ptr_d[i] = (ptr[i] - param.c).mul(cv::Vec2d(1.0 / param.f[0], 1.0 / param.f[1]));
		ptr_d[i][0] -= param.alpha * ptr_d[i][1];
	}

	cv::Matx33d K(param.f[0], param.f[0] * param.alpha, param.c[0],
		0, param.f[1], param.c[1],
		0, 0, 1);

	switch (distortMode)
	{
	case ORIGIN_FISHEYE_CALIB:
		//fisheye::
		cv::fisheye::undistortPoints(distorted, undistorted, cv::Matx33d::eye(), param.k, cv::noArray(), cv::noArray());
		break;
	case THETA_D_FISHEYE_CALIB:
		//fisheye::
		my_cv::fisheye::undistortPoints(distorted, undistorted, cv::Matx33d::eye(), param.k, cv::noArray(), cv::noArray(), cur_fisheye_mode);
		break;
	case RADIUS_D_FISHEYE_CALIB:
		//fisheye_r_d::
		my_cv::fisheye_r_d::undistortPoints_H(imagePoints, undistorted, K, param.k, cur_fisheye_mode);
		break;
	case RADIUS_RD_FISHEYE_CALIB:
		//fisheye_r_rd::
		my_cv::fisheye_r_rd::undistortPoints(distorted, undistorted, cv::Matx33d::eye(), param.k, cv::noArray(), cv::noArray(), cur_fisheye_mode);
		break;
	}

	return undistorted;
}

void my_cv::internal::InitExtrinsics(const cv::Mat& _imagePoints, const cv::Mat& _objectPoints, 
	const IntrinsicParams& param, cv::Mat& omckk, cv::Mat& Tckk, DISTORT_Mode_Fisheye distortMode)
{
	CV_Assert(!_objectPoints.empty() && _objectPoints.type() == CV_64FC3);
	CV_Assert(!_imagePoints.empty() && _imagePoints.type() == CV_64FC2);

	cv::Mat imagePointsNormalized = NormalizePixels(_imagePoints, param, distortMode).reshape(1).t();
	cv::Mat objectPoints = _objectPoints.reshape(1).t();
	cv::Mat objectPointsMean, covObjectPoints;
	cv::Mat Rckk;
	int Np = imagePointsNormalized.cols;
	//坐标归一化
	calcCovarMatrix(objectPoints, covObjectPoints, objectPointsMean, cv::COVAR_NORMAL | cv::COVAR_COLS);
	cv::SVD svd(covObjectPoints);
	cv::Mat R(svd.vt);
	if (cv::norm(R(cv::Rect(2, 0, 1, 2))) < 1e-6)
		R = cv::Mat::eye(3, 3, CV_64FC1);
	if (determinant(R) < 0)//determinant 行列式
		R = -R;
	cv::Mat T = -R * objectPointsMean;
	cv::Mat X_new = R * objectPoints + T * cv::Mat::ones(1, Np, CV_64FC1);  //理想相机坐标系坐标
	//计算基础矩阵H
	cv::Mat H = ComputeHomography(imagePointsNormalized, X_new(cv::Rect(0, 0, X_new.cols, 2)));
	//cv::Mat obj((int)_objectPoints.total(), 1, CV_64FC2), undistorted;
	//const cv::Vec2d* ptr = _objectPoints.ptr<cv::Vec2d>();      //图像像素坐标
	//cv::Vec2d* ptr_d = obj.ptr<cv::Vec2d>();              //
	//for (size_t i = 0; i < _objectPoints.total(); ++i)
	//{
	//	ptr_d[i] = ptr[i];
	//}
	//cv::Mat H = ComputeHomography(imagePointsNormalized, objectPoints(cv::Rect(0, 0, objectPoints.cols, 2)));

	double sc = .5 * (norm(H.col(0)) + norm(H.col(1)));
	H = H / sc;
	cv::Mat u1 = H.col(0).clone();
	double norm_u1 = norm(u1);
	CV_Assert(fabs(norm_u1) > 0);
	u1 = u1 / norm_u1;
	cv::Mat u2 = H.col(1).clone() - u1.dot(H.col(1).clone()) * u1;
	double norm_u2 = norm(u2);
	CV_Assert(fabs(norm_u2) > 0);
	u2 = u2 / norm_u2;

	//cv::Mat u1 = H.col(0).clone();
	//cv::Mat u2 = H.col(1).clone();

	cv::Mat u3 = u1.cross(u2);
	cv::Mat RRR;
	hconcat(u1, u2, RRR);
	hconcat(RRR, u3, RRR);
	Rodrigues(RRR, omckk);
	Rodrigues(omckk, Rckk);
	Tckk = H.col(2).clone();
	Tckk = Tckk + Rckk * T;
	Rckk = Rckk * R;
	Rodrigues(Rckk, omckk);
}

void my_cv::internal::CalibrateExtrinsics(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
	const IntrinsicParams& param, const int check_cond,
	const double thresh_cond, cv::InputOutputArray omc, cv::InputOutputArray Tc, DISTORT_Mode_Fisheye distortMode)
{
	CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
	CV_Assert(!imagePoints.empty() && (imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2));
	CV_Assert(omc.type() == CV_64FC3 || Tc.type() == CV_64FC3);

	if (omc.empty()) omc.create(1, (int)objectPoints.total(), CV_64FC3);
	if (Tc.empty()) Tc.create(1, (int)objectPoints.total(), CV_64FC3);

	const int maxIter = 20;

	for (int image_idx = 0; image_idx < (int)imagePoints.total(); ++image_idx)
	{
		cv::Mat omckk, Tckk, JJ_kk;
		cv::Mat image, object;

		objectPoints.getMat(image_idx).convertTo(object, CV_64FC3);
		imagePoints.getMat(image_idx).convertTo(image, CV_64FC2);

		bool imT = image.rows < image.cols;
		bool obT = object.rows < object.cols;

		InitExtrinsics(imT ? image.t() : image, obT ? object.t() : object, 
			param, omckk, Tckk, distortMode);

		ComputeExtrinsicRefine(!imT ? image.t() : image, !obT ? object.t() : object, 
			omckk, Tckk, JJ_kk, maxIter, param, thresh_cond, distortMode);
		if (check_cond)
		{
			cv::SVD svd(JJ_kk, cv::SVD::NO_UV);
			if (svd.w.at<double>(0) / svd.w.at<double>((int)svd.w.total() - 1) > thresh_cond)
				CV_Error(cv::Error::StsInternal, cv::format("CALIB_CHECK_COND - Ill-conditioned matrix for input array %d", image_idx));
		}
		omckk.reshape(3, 1).copyTo(omc.getMat().col(image_idx));
		Tckk.reshape(3, 1).copyTo(Tc.getMat().col(image_idx));
	}
}

void my_cv::internal::ComputeJacobians(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
	const IntrinsicParams& param, cv::InputArray omc, cv::InputArray Tc,
	const int& check_cond, const double& thresh_cond, cv::Mat& JJ2, cv::Mat& ex3, DISTORT_Mode_Fisheye distortMode)
{
	CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
	CV_Assert(!imagePoints.empty() && (imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2));

	CV_Assert(!omc.empty() && omc.type() == CV_64FC3);
	CV_Assert(!Tc.empty() && Tc.type() == CV_64FC3);

	int n = (int)objectPoints.total();

	JJ2 = cv::Mat::zeros(11 + 6 * n, 11 + 6 * n, CV_64FC1);
	ex3 = cv::Mat::zeros(11 + 6 * n, 1, CV_64FC1);

	if (distortMode != RADIUS_RD_FISHEYE_CALIB)
	{
		for (int image_idx = 0; image_idx < n; ++image_idx)
		{
			cv::Mat image, object;
			objectPoints.getMat(image_idx).convertTo(object, CV_64FC3);
			imagePoints.getMat(image_idx).convertTo(image, CV_64FC2);

			bool imT = image.rows < image.cols;
			cv::Mat om(omc.getMat().col(image_idx)), T(Tc.getMat().col(image_idx));

			std::vector<cv::Point2d> x;
			cv::Mat jacobians;
			projectPoints(object, x, om, T, param, jacobians, distortMode);
			cv::Mat exkk = (imT ? image.t() : image) - cv::Mat(x);

			cv::Mat A(jacobians.rows, 11, CV_64FC1);
			jacobians.colRange(0, 4).copyTo(A.colRange(0, 4));//f,c
			jacobians.col(16).copyTo(A.col(4));//alpha
			jacobians.colRange(4, 10).copyTo(A.colRange(5, 11));//k

			A = A.t();

			cv::Mat B = jacobians.colRange(10, 16).clone();
			B = B.t();

			JJ2(cv::Rect(0, 0, 11, 11)) += A * A.t();
			JJ2(cv::Rect(11 + 6 * image_idx, 11 + 6 * image_idx, 6, 6)) = B * B.t();

			JJ2(cv::Rect(11 + 6 * image_idx, 0, 6, 11)) = A * B.t();
			JJ2(cv::Rect(0, 11 + 6 * image_idx, 11, 6)) = JJ2(cv::Rect(11 + 6 * image_idx, 0, 6, 11)).t();

			ex3.rowRange(0, 11) += A * exkk.reshape(1, 2 * exkk.rows);
			ex3.rowRange(11 + 6 * image_idx, 11 + 6 * (image_idx + 1)) = B * exkk.reshape(1, 2 * exkk.rows);

			if (check_cond)
			{
				cv::Mat JJ_kk = B.t();
				cv::SVD svd(JJ_kk, cv::SVD::NO_UV);
				CV_Assert(svd.w.at<double>(0) / svd.w.at<double>(svd.w.rows - 1) < thresh_cond);
			}
		}
	}
	else
	{
		for (int image_idx = 0; image_idx < n; ++image_idx)
		{
			cv::Mat image, object;
			objectPoints.getMat(image_idx).convertTo(object, CV_64FC3);
			imagePoints.getMat(image_idx).convertTo(image, CV_64FC2);

			bool objT = object.rows < object.cols;
			cv::Mat om(omc.getMat().col(image_idx)), T(Tc.getMat().col(image_idx));

			std::vector<cv::Point3d> x;
			cv::Mat jacobians;
			projectPoints(x, image, om, T, param, jacobians, distortMode);
			cv::Mat exkk = (objT ? object.t() : object) - cv::Mat(x);

			cv::Mat A(jacobians.rows, 11, CV_64FC1);
			jacobians.colRange(0, 4).copyTo(A.colRange(0, 4));//f,c
			jacobians.col(14).copyTo(A.col(4));//alpha
			jacobians.colRange(4, 10).copyTo(A.colRange(5, 11));//k

			A = A.t();

			cv::Mat B = jacobians.colRange(10, 16).clone();
			B = B.t();

			JJ2(cv::Rect(0, 0, 11, 11)) += A * A.t();
			JJ2(cv::Rect(11 + 6 * image_idx, 11 + 6 * image_idx, 6, 6)) = B * B.t();

			JJ2(cv::Rect(11 + 6 * image_idx, 0, 6, 11)) = A * B.t();
			JJ2(cv::Rect(0, 11 + 6 * image_idx, 11, 6)) = JJ2(cv::Rect(11 + 6 * image_idx, 0, 6, 11)).t();

			ex3.rowRange(0, 11) += A * exkk.reshape(1, 3 * exkk.rows);
			ex3.rowRange(11 + 6 * image_idx, 11 + 6 * (image_idx + 1)) = B * exkk.reshape(1, 3 * exkk.rows);

			if (check_cond)
			{
				cv::Mat JJ_kk = B.t();
				cv::SVD svd(JJ_kk, cv::SVD::NO_UV);
				CV_Assert(svd.w.at<double>(0) / svd.w.at<double>(svd.w.rows - 1) < thresh_cond);
			}
		}
	}

	std::vector<uchar> idxs(param.isEstimate);
	idxs.insert(idxs.end(), 6 * n, 1);

	subMatrix(JJ2, JJ2, idxs, idxs);
	subMatrix(ex3, ex3, std::vector<uchar>(1, 1), idxs);
}

void my_cv::internal::EstimateUncertainties(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
	const IntrinsicParams& params, cv::InputArray omc, cv::InputArray Tc,
	IntrinsicParams& errors, cv::Vec2d& std_err, double thresh_cond, int check_cond, double& rms,
	DISTORT_Mode_Fisheye distortMode)
{
	CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
	CV_Assert(!imagePoints.empty() && (imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2));

	CV_Assert(!omc.empty() && omc.type() == CV_64FC3);
	CV_Assert(!Tc.empty() && Tc.type() == CV_64FC3);

	if (distortMode != RADIUS_RD_FISHEYE_CALIB)
	{
		int total_ex = 0;
		for (int image_idx = 0; image_idx < (int)objectPoints.total(); ++image_idx)
		{
			total_ex += (int)objectPoints.getMat(image_idx).total();
		}
		cv::Mat ex(total_ex, 1, CV_64FC2);
		int insert_idx = 0;
		for (int image_idx = 0; image_idx < (int)objectPoints.total(); ++image_idx)
		{
			cv::Mat image, object;
			objectPoints.getMat(image_idx).convertTo(object, CV_64FC3);
			imagePoints.getMat(image_idx).convertTo(image, CV_64FC2);

			bool imT = image.rows < image.cols;

			cv::Mat om(omc.getMat().col(image_idx)), T(Tc.getMat().col(image_idx));

			std::vector<cv::Point2d> x;
			projectPoints(object, x, om, T, params, cv::noArray(), distortMode);
			cv::Mat ex_ = (imT ? image.t() : image) - cv::Mat(x);
			ex_.copyTo(ex.rowRange(insert_idx, insert_idx + ex_.rows));
			insert_idx += ex_.rows;
		}

		meanStdDev(ex, cv::noArray(), std_err);
		std_err *= sqrt((double)ex.total() / ((double)ex.total() - 1.0));

		cv::Vec<double, 1> sigma_x;
		meanStdDev(ex.reshape(1, 1), cv::noArray(), sigma_x);
		sigma_x *= sqrt(2.0 * (double)ex.total() / (2.0 * (double)ex.total() - 1.0));

		cv::Mat JJ2, ex3;
		ComputeJacobians(objectPoints, imagePoints, params, omc, Tc, check_cond, thresh_cond, JJ2, ex3, distortMode);

		sqrt(JJ2.inv(), JJ2);

		errors = 3 * sigma_x(0) * JJ2.diag();
		rms = sqrt(norm(ex, cv::NORM_L2SQR) / ex.total());
	}
}

void my_cv::internal::EstimateUncertainties_rd(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
	const IntrinsicParams& params, cv::InputArray omc, cv::InputArray Tc, IntrinsicParams& errors, cv::Vec3d& std_err,
	double thresh_cond, int check_cond, double& rms, DISTORT_Mode_Fisheye distortMode)
{
	CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
	CV_Assert(!imagePoints.empty() && (imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2));

	CV_Assert(!omc.empty() && omc.type() == CV_64FC3);
	CV_Assert(!Tc.empty() && Tc.type() == CV_64FC3);

	int total_ex = 0;
	for (int image_idx = 0; image_idx < (int)objectPoints.total(); ++image_idx)
	{
		total_ex += (int)objectPoints.getMat(image_idx).total();
	}
	cv::Mat ex(total_ex, 1, CV_64FC3);
	int insert_idx = 0;
	for (int image_idx = 0; image_idx < (int)objectPoints.total(); ++image_idx)
	{
		cv::Mat image, object;
		objectPoints.getMat(image_idx).convertTo(object, CV_64FC3);
		imagePoints.getMat(image_idx).convertTo(image, CV_64FC2);

		bool objT = object.rows < object.cols;

		cv::Mat om(omc.getMat().col(image_idx)), T(Tc.getMat().col(image_idx));

		std::vector<cv::Point3d> x;
		projectPoints(x, image, om, T, params, cv::noArray(), distortMode);
		cv::Mat ex_ = (objT ? object.t() : object) - cv::Mat(x);
		ex_.copyTo(ex.rowRange(insert_idx, insert_idx + ex_.rows));
		insert_idx += ex_.rows;
	}

	meanStdDev(ex, cv::noArray(), std_err);
	std_err *= sqrt((double)ex.total() / ((double)ex.total() - 1.0));

	cv::Vec<double, 1> sigma_x;
	meanStdDev(ex.reshape(1, 1), cv::noArray(), sigma_x);
	sigma_x *= sqrt(3.0 * (double)ex.total() / (3.0 * (double)ex.total() - 1.0));

	cv::Mat JJ2, ex3;
	ComputeJacobians(objectPoints, imagePoints, params, omc, Tc, check_cond, thresh_cond, JJ2, ex3, distortMode);

	sqrt(JJ2.inv(), JJ2);

	errors = 3 * sigma_x(0) * JJ2.diag();
	rms = sqrt(norm(ex, cv::NORM_L2SQR) / ex.total());
}

void my_cv::internal::dAB(cv::InputArray A, cv::InputArray B, cv::OutputArray dABdA, cv::OutputArray dABdB)
{
	CV_Assert(A.getMat().cols == B.getMat().rows);
	CV_Assert(A.type() == CV_64FC1 && B.type() == CV_64FC1);

	int p = A.getMat().rows;
	int n = A.getMat().cols;
	int q = B.getMat().cols;

	dABdA.create(p * q, p * n, CV_64FC1);
	dABdB.create(p * q, q * n, CV_64FC1);

	dABdA.getMat() = cv::Mat::zeros(p * q, p * n, CV_64FC1);
	dABdB.getMat() = cv::Mat::zeros(p * q, q * n, CV_64FC1);

	for (int i = 0; i < q; ++i)
	{
		for (int j = 0; j < p; ++j)
		{
			int ij = j + i * p;
			for (int k = 0; k < n; ++k)
			{
				int kj = j + k * p;
				dABdA.getMat().at<double>(ij, kj) = B.getMat().at<double>(k, i);
			}
		}
	}

	for (int i = 0; i < q; ++i)
	{
		A.getMat().copyTo(dABdB.getMat().rowRange(i * p, i * p + p).colRange(i * n, i * n + n));
	}
}

void my_cv::internal::JRodriguesMatlab(const cv::Mat& src, cv::Mat& dst)
{
	cv::Mat tmp(src.cols, src.rows, src.type());
	if (src.rows == 9)
	{
		cv::Mat(src.row(0).t()).copyTo(tmp.col(0));
		cv::Mat(src.row(1).t()).copyTo(tmp.col(3));
		cv::Mat(src.row(2).t()).copyTo(tmp.col(6));
		cv::Mat(src.row(3).t()).copyTo(tmp.col(1));
		cv::Mat(src.row(4).t()).copyTo(tmp.col(4));
		cv::Mat(src.row(5).t()).copyTo(tmp.col(7));
		cv::Mat(src.row(6).t()).copyTo(tmp.col(2));
		cv::Mat(src.row(7).t()).copyTo(tmp.col(5));
		cv::Mat(src.row(8).t()).copyTo(tmp.col(8));
	}
	else
	{
		cv::Mat(src.col(0).t()).copyTo(tmp.row(0));
		cv::Mat(src.col(1).t()).copyTo(tmp.row(3));
		cv::Mat(src.col(2).t()).copyTo(tmp.row(6));
		cv::Mat(src.col(3).t()).copyTo(tmp.row(1));
		cv::Mat(src.col(4).t()).copyTo(tmp.row(4));
		cv::Mat(src.col(5).t()).copyTo(tmp.row(7));
		cv::Mat(src.col(6).t()).copyTo(tmp.row(2));
		cv::Mat(src.col(7).t()).copyTo(tmp.row(5));
		cv::Mat(src.col(8).t()).copyTo(tmp.row(8));
	}
	dst = tmp.clone();
}

void my_cv::internal::compose_motion(cv::InputArray _om1, cv::InputArray _T1, cv::InputArray _om2, cv::InputArray _T2,
	cv::Mat& om3, cv::Mat& T3, cv::Mat& dom3dom1, cv::Mat& dom3dT1, cv::Mat& dom3dom2,
	cv::Mat& dom3dT2, cv::Mat& dT3dom1, cv::Mat& dT3dT1, cv::Mat& dT3dom2, cv::Mat& dT3dT2)
{
	cv::Mat om1 = _om1.getMat();
	cv::Mat om2 = _om2.getMat();
	cv::Mat T1 = _T1.getMat().reshape(1, 3);
	cv::Mat T2 = _T2.getMat().reshape(1, 3);

	//% Rotations:
	cv::Mat R1, R2, R3, dR1dom1(9, 3, CV_64FC1), dR2dom2;
	Rodrigues(om1, R1, dR1dom1);
	Rodrigues(om2, R2, dR2dom2);
	JRodriguesMatlab(dR1dom1, dR1dom1);
	JRodriguesMatlab(dR2dom2, dR2dom2);
	R3 = R2 * R1;
	cv::Mat dR3dR2, dR3dR1;
	dAB(R2, R1, dR3dR2, dR3dR1);
	cv::Mat dom3dR3;
	Rodrigues(R3, om3, dom3dR3);
	JRodriguesMatlab(dom3dR3, dom3dR3);
	dom3dom1 = dom3dR3 * dR3dR1 * dR1dom1;
	dom3dom2 = dom3dR3 * dR3dR2 * dR2dom2;
	dom3dT1 = cv::Mat::zeros(3, 3, CV_64FC1);
	dom3dT2 = cv::Mat::zeros(3, 3, CV_64FC1);

	//% Translations:
	cv::Mat T3t = R2 * T1;
	cv::Mat dT3tdR2, dT3tdT1;
	dAB(R2, T1, dT3tdR2, dT3tdT1);
	cv::Mat dT3tdom2 = dT3tdR2 * dR2dom2;
	T3 = T3t + T2;
	dT3dT1 = dT3tdT1;
	dT3dT2 = cv::Mat::eye(3, 3, CV_64FC1);
	dT3dom2 = dT3tdom2;
	dT3dom1 = cv::Mat::zeros(3, 3, CV_64FC1);
}

double my_cv::internal::median(const cv::Mat& row)
{
	CV_Assert(row.type() == CV_64FC1);
	CV_Assert(!row.empty() && row.rows == 1);
	cv::Mat tmp = row.clone();
	sort(tmp, tmp, 0);
	if ((int)tmp.total() % 2) return tmp.at<double>((int)tmp.total() / 2);
	else return 0.5 *(tmp.at<double>((int)tmp.total() / 2) + tmp.at<double>((int)tmp.total() / 2 - 1));
}

cv::Vec3d my_cv::internal::median3d(cv::InputArray m)
{
	CV_Assert(m.depth() == CV_64F && m.getMat().rows == 1);
	cv::Mat M = cv::Mat(m.getMat().t()).reshape(1).t();
	return cv::Vec3d(median(M.row(0)), median(M.row(1)), median(M.row(2)));
}

double getR(double theta, camMode mode)
{
	double r;
	switch (mode)
	{
	case STEREOGRAPHIC:
		r = 2 * tan(theta / 2.0);
		break;
	case EQUIDISTANCE:
		r = theta;
		break;
	case EQUISOLID:
		r = 2 * sin(theta / 2.0);
		break;
	case ORTHOGONAL:
		r = sin(theta);
		break;
	case IDEAL_PERSPECTIVE:
		r = tan(theta);
		break;
	}
	return r;
}

double getTheta(double r, camMode mode)
{
	double theta;
	switch (mode)
	{
	case STEREOGRAPHIC:
		theta = 2 * atan(r / 2.0);
		break;
	case EQUIDISTANCE:
		theta = r;
		break;
	case EQUISOLID:
		theta = 2 * asin(r / 2.0);
		break;
	case ORTHOGONAL:
		theta = asin(r);
		break;
	case IDEAL_PERSPECTIVE:
		theta = atan(r);
		break;
	}
	return theta;
}

double get_drdtheta(double theta, camMode mode)
{
	double drdtheta;
	switch (mode)
	{
	case STEREOGRAPHIC:
	{
		double temp = cos(theta / 2.0);
		temp = temp * temp;
		drdtheta = 1.0 / temp;
	}
	break;
	case EQUIDISTANCE:
		drdtheta = 1.0;
		break;
	case EQUISOLID:
		drdtheta = cos(theta / 2.0);
		break;
	case ORTHOGONAL:
		drdtheta = cos(theta);
		break;
	case IDEAL_PERSPECTIVE:
	{
		double temp_ = cos(theta);
		temp_ = temp_ * temp_;
		drdtheta = 1.0 / temp_;
	}
	break;
	}

	return drdtheta;
}

double get_dthetadr(double r, camMode mode)
{
	double r2 = r * r;
	double dthetadr;
	switch (mode)
	{
	case STEREOGRAPHIC:
		dthetadr = 4.0 / (4.0 + r2);
		break;
	case EQUIDISTANCE:
		dthetadr = 1.0;
		break;
	case EQUISOLID:
		dthetadr = 2.0 / sqrt(4.0 - r2);
		break;
	case ORTHOGONAL:
		dthetadr = 1.0 / sqrt(1.0 - r2);
		break;
	case IDEAL_PERSPECTIVE:
		dthetadr = 1.0 / (1.0 + r2);
		break;
	}
	return dthetadr;
}
