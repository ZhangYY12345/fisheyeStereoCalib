#include "fisheyeCalib_try.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

camMode cur_fisheye_mode = EQUISOLID;

namespace my_cv {
	namespace
	{
		struct JacobianRow
		{
			cv::Vec2d df, dc;
			cv::Vec4d dk;
			cv::Vec3d dom, dT;
			double dalpha;
		};

		void subMatrix(const cv::Mat& src, cv::Mat& dst, const std::vector<uchar>& cols, const std::vector<uchar>& rows);
	}
}


namespace my_cv {
	namespace {
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
}

my_cv::internal::IntrinsicParams::IntrinsicParams() :
	f(cv::Vec2d::all(0)), c(cv::Vec2d::all(0)), k(cv::Vec4d::all(0)), alpha(0), isEstimate(9, 0)
{
}

my_cv::internal::IntrinsicParams::IntrinsicParams(cv::Vec2d _f, cv::Vec2d _c, cv::Vec4d _k, double _alpha) :
	f(_f), c(_c), k(_k), alpha(_alpha), isEstimate(9, 0)
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

	return *this;
}

void my_cv::internal::IntrinsicParams::Init(const cv::Vec2d& _f, const cv::Vec2d& _c, const cv::Vec4d& _k, const double& _alpha)
{
	this->c = _c;
	this->f = _f;
	this->k = _k;
	this->alpha = _alpha;
}

void my_cv::internal::projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints,
	cv::InputArray _rvec, cv::InputArray _tvec,
	const IntrinsicParams& param, cv::OutputArray jacobian)
{
	CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
	cv::Matx33d K(param.f[0], param.f[0] * param.alpha, param.c[0],
		0, param.f[1], param.c[1],
		0, 0, 1);
	my_cv::fisheye::projectPoints(objectPoints, imagePoints, _rvec, _tvec, K, param.k, param.alpha, jacobian, cur_fisheye_mode);//////
}

void my_cv::internal::ComputeExtrinsicRefine(const cv::Mat& imagePoints, const cv::Mat& objectPoints, cv::Mat& rvec,
	cv::Mat&  tvec, cv::Mat& J, const int MaxIter,
	const IntrinsicParams& param, const double thresh_cond)
{
	CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
	CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);
	CV_Assert(rvec.total() > 2 && tvec.total() > 2);
	cv::Vec6d extrinsics(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
		tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
	double change = 1;
	int iter = 0;

	while (change > 1e-10 && iter < MaxIter)
	{
		std::vector<cv::Point2d> x;
		cv::Mat jacobians;
		projectPoints(objectPoints, x, rvec, tvec, param, jacobians);

		cv::Mat ex = imagePoints - cv::Mat(x).t();
		ex = ex.reshape(1, 2);

		J = jacobians.colRange(8, 14).clone();

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

cv::Mat my_cv::internal::NormalizePixels(const cv::Mat& imagePoints, const IntrinsicParams& param)
{
	CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);

	cv::Mat distorted((int)imagePoints.total(), 1, CV_64FC2), undistorted;
	const cv::Vec2d* ptr = imagePoints.ptr<cv::Vec2d>();      //图像像素坐标
	cv::Vec2d* ptr_d = distorted.ptr<cv::Vec2d>();              //归一化的相机坐标系坐标
	for (size_t i = 0; i < imagePoints.total(); ++i)
	{
		ptr_d[i] = (ptr[i] - param.c).mul(cv::Vec2d(1.0 / param.f[0], 1.0 / param.f[1]));
		ptr_d[i][0] -= param.alpha * ptr_d[i][1];
	}
	my_cv::fisheye::undistortPoints(distorted, undistorted, cv::Matx33d::eye(), param.k, cv::noArray(), cv::noArray(), cur_fisheye_mode);
	return undistorted;
}

void my_cv::internal::InitExtrinsics(const cv::Mat& _imagePoints, const cv::Mat& _objectPoints, const IntrinsicParams& param, cv::Mat& omckk, cv::Mat& Tckk)
{
	CV_Assert(!_objectPoints.empty() && _objectPoints.type() == CV_64FC3);
	CV_Assert(!_imagePoints.empty() && _imagePoints.type() == CV_64FC2);

	cv::Mat imagePointsNormalized = NormalizePixels(_imagePoints, param).reshape(1).t();
	cv::Mat objectPoints = _objectPoints.reshape(1).t();
	cv::Mat objectPointsMean, covObjectPoints;
	cv::Mat Rckk;
	int Np = imagePointsNormalized.cols;
	calcCovarMatrix(objectPoints, covObjectPoints, objectPointsMean, cv::COVAR_NORMAL | cv::COVAR_COLS);
	cv::SVD svd(covObjectPoints);
	cv::Mat R(svd.vt);
	if (cv::norm(R(cv::Rect(2, 0, 1, 2))) < 1e-6)
		R = cv::Mat::eye(3, 3, CV_64FC1);
	if (determinant(R) < 0)//determinant 行列式
		R = -R;
	cv::Mat T = -R * objectPointsMean;
	cv::Mat X_new = R * objectPoints + T * cv::Mat::ones(1, Np, CV_64FC1);  //理想相机坐标系坐标
	cv::Mat H = ComputeHomography(imagePointsNormalized, X_new(cv::Rect(0, 0, X_new.cols, 2)));
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
	const double thresh_cond, cv::InputOutputArray omc, cv::InputOutputArray Tc)
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

		InitExtrinsics(imT ? image.t() : image, obT ? object.t() : object, param, omckk, Tckk);

		ComputeExtrinsicRefine(!imT ? image.t() : image, !obT ? object.t() : object, omckk, Tckk, JJ_kk, maxIter, param, thresh_cond);
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
	const int& check_cond, const double& thresh_cond, cv::Mat& JJ2, cv::Mat& ex3)
{
	CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
	CV_Assert(!imagePoints.empty() && (imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2));

	CV_Assert(!omc.empty() && omc.type() == CV_64FC3);
	CV_Assert(!Tc.empty() && Tc.type() == CV_64FC3);

	int n = (int)objectPoints.total();

	JJ2 = cv::Mat::zeros(9 + 6 * n, 9 + 6 * n, CV_64FC1);
	ex3 = cv::Mat::zeros(9 + 6 * n, 1, CV_64FC1);

	for (int image_idx = 0; image_idx < n; ++image_idx)
	{
		cv::Mat image, object;
		objectPoints.getMat(image_idx).convertTo(object, CV_64FC3);
		imagePoints.getMat(image_idx).convertTo(image, CV_64FC2);

		bool imT = image.rows < image.cols;
		cv::Mat om(omc.getMat().col(image_idx)), T(Tc.getMat().col(image_idx));

		std::vector<cv::Point2d> x;
		cv::Mat jacobians;
		projectPoints(object, x, om, T, param, jacobians);
		cv::Mat exkk = (imT ? image.t() : image) - cv::Mat(x);

		cv::Mat A(jacobians.rows, 9, CV_64FC1);
		jacobians.colRange(0, 4).copyTo(A.colRange(0, 4));
		jacobians.col(14).copyTo(A.col(4));
		jacobians.colRange(4, 8).copyTo(A.colRange(5, 9));

		A = A.t();

		cv::Mat B = jacobians.colRange(8, 14).clone();
		B = B.t();

		JJ2(cv::Rect(0, 0, 9, 9)) += A * A.t();
		JJ2(cv::Rect(9 + 6 * image_idx, 9 + 6 * image_idx, 6, 6)) = B * B.t();

		JJ2(cv::Rect(9 + 6 * image_idx, 0, 6, 9)) = A * B.t();
		JJ2(cv::Rect(0, 9 + 6 * image_idx, 9, 6)) = JJ2(cv::Rect(9 + 6 * image_idx, 0, 6, 9)).t();

		ex3.rowRange(0, 9) += A * exkk.reshape(1, 2 * exkk.rows);
		ex3.rowRange(9 + 6 * image_idx, 9 + 6 * (image_idx + 1)) = B * exkk.reshape(1, 2 * exkk.rows);

		if (check_cond)
		{
			cv::Mat JJ_kk = B.t();
			cv::SVD svd(JJ_kk, cv::SVD::NO_UV);
			CV_Assert(svd.w.at<double>(0) / svd.w.at<double>(svd.w.rows - 1) < thresh_cond);
		}
	}

	std::vector<uchar> idxs(param.isEstimate);
	idxs.insert(idxs.end(), 6 * n, 1);

	subMatrix(JJ2, JJ2, idxs, idxs);
	subMatrix(ex3, ex3, std::vector<uchar>(1, 1), idxs);
}

void my_cv::internal::EstimateUncertainties(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints,
	const IntrinsicParams& params, cv::InputArray omc, cv::InputArray Tc,
	IntrinsicParams& errors, cv::Vec2d& std_err, double thresh_cond, int check_cond, double& rms)
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
		projectPoints(object, x, om, T, params, cv::noArray());
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
	ComputeJacobians(objectPoints, imagePoints, params, omc, Tc, check_cond, thresh_cond, JJ2, ex3);

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



//

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::projectPoints

void my_cv::fisheye::projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints, const cv::Affine3d& affine,
    cv::InputArray K, cv::InputArray D, double alpha, cv::OutputArray jacobian, camMode mode)
{
    projectPoints(objectPoints, imagePoints, affine.rvec(), affine.translation(), K, D, alpha, jacobian, mode);
}

void my_cv::fisheye::projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints, cv::InputArray _rvec,
        cv::InputArray _tvec, cv::InputArray _K, cv::InputArray _D, double alpha, cv::OutputArray jacobian, camMode mode)
{
    // will support only 3-channel data now for points
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
    imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));
    size_t n = objectPoints.total();

    CV_Assert(_rvec.total() * _rvec.channels() == 3 && (_rvec.depth() == CV_32F || _rvec.depth() == CV_64F));
    CV_Assert(_tvec.total() * _tvec.channels() == 3 && (_tvec.depth() == CV_32F || _tvec.depth() == CV_64F));
    CV_Assert(_tvec.getMat().isContinuous() && _rvec.getMat().isContinuous());

    cv::Vec3d om = _rvec.depth() == CV_32F ? (cv::Vec3d)*_rvec.getMat().ptr<cv::Vec3f>() : *_rvec.getMat().ptr<cv::Vec3d>();
    cv::Vec3d T  = _tvec.depth() == CV_32F ? (cv::Vec3d)*_tvec.getMat().ptr<cv::Vec3f>() : *_tvec.getMat().ptr<cv::Vec3d>();

    CV_Assert(_K.size() == cv::Size(3,3) && (_K.type() == CV_32F || _K.type() == CV_64F) && _D.type() == _K.type() && _D.total() == 4);

    cv::Vec2d f, c;
    if (_K.depth() == CV_32F)
    {

        cv::Matx33f K = _K.getMat();
        f = cv::Vec2f(K(0, 0), K(1, 1));
        c = cv::Vec2f(K(0, 2), K(1, 2));
    }
    else
    {
        cv::Matx33d K = _K.getMat();
        f = cv::Vec2d(K(0, 0), K(1, 1));
        c = cv::Vec2d(K(0, 2), K(1, 2));
    }

    cv::Vec4d k = _D.depth() == CV_32F ? (cv::Vec4d)*_D.getMat().ptr<cv::Vec4f>(): *_D.getMat().ptr<cv::Vec4d>();

    const bool isJacobianNeeded = jacobian.needed();
    JacobianRow *Jn = 0;
    if (isJacobianNeeded)
    {
        int nvars = 2 + 2 + 1 + 4 + 3 + 3; // f, c, alpha, k, om, T,
        jacobian.create(2*(int)n, nvars, CV_64F);
        Jn = jacobian.getMat().ptr<JacobianRow>(0);
    }

    cv::Matx33d R;
    cv::Matx<double, 3, 9> dRdom;
    Rodrigues(om, R, dRdom);
    cv::Affine3d aff(om, T);

    const cv::Vec3f* Xf = objectPoints.getMat().ptr<cv::Vec3f>();
    const cv::Vec3d* Xd = objectPoints.getMat().ptr<cv::Vec3d>();
    cv::Vec2f *xpf = imagePoints.getMat().ptr<cv::Vec2f>();
    cv::Vec2d *xpd = imagePoints.getMat().ptr<cv::Vec2d>();

    for(size_t i = 0; i < n; ++i)
    {
        cv::Vec3d Xi = objectPoints.depth() == CV_32F ? (cv::Vec3d)Xf[i] : Xd[i];
        cv::Vec3d Y = aff*Xi;
        if (fabs(Y[2]) < DBL_MIN)
            Y[2] = 1;
        cv::Vec2d x(Y[0]/Y[2], Y[1]/Y[2]);

        double r2 = x.dot(x);
        double r = std::sqrt(r2);

        // Angle of the incoming ray:
		double theta;
    	switch(mode)
    	{
		case STEREOGRAPHIC:
			theta = 2 * atan(r / 2);		//r = 2f * tan(theta / 2)
			break;
		case EQUIDISTANCE:
			theta = r;						// r = f * theta
			break;
		case EQUISOLID:
			theta = 2 * asin(r / 2);		// r = 2f * sin(theta / 2)
			break;
		case ORTHOGONAL:
			theta = asin(r);				// r = f * sin(theta)
			break;
		default:
			theta = atan(r);				//r = f * tan(theta)
    	}
    						
		

        double theta2 = theta*theta, theta3 = theta2*theta, theta4 = theta2*theta2, theta5 = theta4*theta,
                theta6 = theta3*theta3, theta7 = theta6*theta, theta8 = theta4*theta4, theta9 = theta8*theta;

        double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 + k[3]*theta9;

        double inv_r = r > 1e-8 ? 1.0/r : 1;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1;

        cv::Vec2d xd1 = x * cdist;
        cv::Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
        cv::Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);

        if (objectPoints.depth() == CV_32F)
            xpf[i] = final_point;
        else
            xpd[i] = final_point;

        if (isJacobianNeeded)
        {
            //cv::Vec3d Xi = pdepth == CV_32F ? (cv::Vec3d)Xf[i] : Xd[i];
            //cv::Vec3d Y = aff*Xi;
            double dYdR[] = { Xi[0], Xi[1], Xi[2], 0, 0, 0, 0, 0, 0,
                              0, 0, 0, Xi[0], Xi[1], Xi[2], 0, 0, 0,
                              0, 0, 0, 0, 0, 0, Xi[0], Xi[1], Xi[2] };

            cv::Matx33d dYdom_data = cv::Matx<double, 3, 9>(dYdR) * dRdom.t();
            const cv::Vec3d *dYdom = (cv::Vec3d*)dYdom_data.val;

            cv::Matx33d dYdT_data = cv::Matx33d::eye();
            const cv::Vec3d *dYdT = (cv::Vec3d*)dYdT_data.val;

            //cv::Vec2d x(Y[0]/Y[2], Y[1]/Y[2]);
            cv::Vec3d dxdom[2];
            dxdom[0] = (1.0/Y[2]) * dYdom[0] - x[0]/Y[2] * dYdom[2];
            dxdom[1] = (1.0/Y[2]) * dYdom[1] - x[1]/Y[2] * dYdom[2];

            cv::Vec3d dxdT[2];
            dxdT[0]  = (1.0/Y[2]) * dYdT[0] - x[0]/Y[2] * dYdT[2];
            dxdT[1]  = (1.0/Y[2]) * dYdT[1] - x[1]/Y[2] * dYdT[2];

            //double r2 = x.dot(x);
            cv::Vec3d dr2dom = 2 * x[0] * dxdom[0] + 2 * x[1] * dxdom[1];
            cv::Vec3d dr2dT  = 2 * x[0] *  dxdT[0] + 2 * x[1] *  dxdT[1];

            //double r = std::sqrt(r2);
            double drdr2 = r > 1e-8 ? 1.0/(2*r) : 1;
            cv::Vec3d drdom = drdr2 * dr2dom;
            cv::Vec3d drdT  = drdr2 * dr2dT;

            // Angle of the incoming ray:
            //double theta = atan(r);
            double dthetadr = 1.0/(1+r2);
            cv::Vec3d dthetadom = dthetadr * drdom;
            cv::Vec3d dthetadT  = dthetadr *  drdT;

            //double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 + k[3]*theta9;
            double dtheta_ddtheta = 1 + 3*k[0]*theta2 + 5*k[1]*theta4 + 7*k[2]*theta6 + 9*k[3]*theta8;
            cv::Vec3d dtheta_ddom = dtheta_ddtheta * dthetadom;
            cv::Vec3d dtheta_ddT  = dtheta_ddtheta * dthetadT;
            cv::Vec4d dtheta_ddk  = cv::Vec4d(theta3, theta5, theta7, theta9);

            //double inv_r = r > 1e-8 ? 1.0/r : 1;
            //double cdist = r > 1e-8 ? theta_d / r : 1;
            cv::Vec3d dcdistdom = inv_r * (dtheta_ddom - cdist*drdom);
            cv::Vec3d dcdistdT  = inv_r * (dtheta_ddT  - cdist*drdT);
            cv::Vec4d dcdistdk  = inv_r *  dtheta_ddk;

            //cv::Vec2d xd1 = x * cdist;
            cv::Vec4d dxd1dk[2];
            cv::Vec3d dxd1dom[2], dxd1dT[2];
            dxd1dom[0] = x[0] * dcdistdom + cdist * dxdom[0];
            dxd1dom[1] = x[1] * dcdistdom + cdist * dxdom[1];
            dxd1dT[0]  = x[0] * dcdistdT  + cdist * dxdT[0];
            dxd1dT[1]  = x[1] * dcdistdT  + cdist * dxdT[1];
            dxd1dk[0]  = x[0] * dcdistdk;
            dxd1dk[1]  = x[1] * dcdistdk;

            //cv::Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
            cv::Vec4d dxd3dk[2];
            cv::Vec3d dxd3dom[2], dxd3dT[2];
            dxd3dom[0] = dxd1dom[0] + alpha * dxd1dom[1];
            dxd3dom[1] = dxd1dom[1];
            dxd3dT[0]  = dxd1dT[0]  + alpha * dxd1dT[1];
            dxd3dT[1]  = dxd1dT[1];
            dxd3dk[0]  = dxd1dk[0]  + alpha * dxd1dk[1];
            dxd3dk[1]  = dxd1dk[1];

            cv::Vec2d dxd3dalpha(xd1[1], 0);

            //final jacobian
            Jn[0].dom = f[0] * dxd3dom[0];
            Jn[1].dom = f[1] * dxd3dom[1];

            Jn[0].dT = f[0] * dxd3dT[0];
            Jn[1].dT = f[1] * dxd3dT[1];

            Jn[0].dk = f[0] * dxd3dk[0];
            Jn[1].dk = f[1] * dxd3dk[1];

            Jn[0].dalpha = f[0] * dxd3dalpha[0];
            Jn[1].dalpha = 0; //f[1] * dxd3dalpha[1];

            Jn[0].df = cv::Vec2d(xd3[0], 0);
            Jn[1].df = cv::Vec2d(0, xd3[1]);

            Jn[0].dc = cv::Vec2d(1, 0);
            Jn[1].dc = cv::Vec2d(0, 1);

            //step to jacobian rows for next point
            Jn += 2;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::distortPoints

void my_cv::fisheye::distortPoints(cv::InputArray undistorted, cv::OutputArray distorted, cv::InputArray K, cv::InputArray D, double alpha, camMode mode)
{
    // will support only 2-channel data now for points
    CV_Assert(undistorted.type() == CV_32FC2 || undistorted.type() == CV_64FC2);
    distorted.create(undistorted.size(), undistorted.type());
    size_t n = undistorted.total();

    CV_Assert(K.size() == cv::Size(3,3) && (K.type() == CV_32F || K.type() == CV_64F) && D.total() == 4);

    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        cv::Matx33f camMat = K.getMat();
        f = cv::Vec2f(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        cv::Matx33d camMat = K.getMat();
        f = cv::Vec2d(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2d(camMat(0 ,2), camMat(1, 2));
    }

    cv::Vec4d k = D.depth() == CV_32F ? (cv::Vec4d)*D.getMat().ptr<cv::Vec4f>(): *D.getMat().ptr<cv::Vec4d>();

    const cv::Vec2f* Xf = undistorted.getMat().ptr<cv::Vec2f>();
    const cv::Vec2d* Xd = undistorted.getMat().ptr<cv::Vec2d>();
    cv::Vec2f *xpf = distorted.getMat().ptr<cv::Vec2f>();
    cv::Vec2d *xpd = distorted.getMat().ptr<cv::Vec2d>();

    for(size_t i = 0; i < n; ++i)
    {
        cv::Vec2d x = undistorted.depth() == CV_32F ? (cv::Vec2d)Xf[i] : Xd[i];

        double r2 = x.dot(x);
        double r = std::sqrt(r2);

        // Angle of the incoming ray:
        double theta;
		switch (mode)
		{
		case STEREOGRAPHIC:
			theta = 2 * atan(r / 2);		//r = 2f * tan(theta / 2)
			break;
		case EQUIDISTANCE:
			theta = r;						// r = f * theta
			break;
		case EQUISOLID:
			theta = 2 * asin(r / 2);		// r = 2f * sin(theta / 2)
			break;
		case ORTHOGONAL:
			theta = asin(r);				// r = f * sin(theta)
			break;
		default:
			theta = atan(r);				//r = f * tan(theta)
		}

        double theta2 = theta*theta, theta3 = theta2*theta, theta4 = theta2*theta2, theta5 = theta4*theta,
                theta6 = theta3*theta3, theta7 = theta6*theta, theta8 = theta4*theta4, theta9 = theta8*theta;

        double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 + k[3]*theta9;

        double inv_r = r > 1e-8 ? 1.0/r : 1;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1;

        cv::Vec2d xd1 = x * cdist;
        cv::Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
        cv::Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);

        if (undistorted.depth() == CV_32F)
            xpf[i] = final_point;
        else
            xpd[i] = final_point;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::undistortPoints

void my_cv::fisheye::undistortPoints( cv::InputArray distorted, cv::OutputArray undistorted, 
	cv::InputArray K, cv::InputArray D, cv::InputArray R, cv::InputArray P, camMode mode)
{
    // will support only 2-channel data now for points
    CV_Assert(distorted.type() == CV_32FC2 || distorted.type() == CV_64FC2);
    undistorted.create(distorted.size(), distorted.type());

    CV_Assert(P.empty() || P.size() == cv::Size(3, 3) || P.size() == cv::Size(4, 3));
    CV_Assert(R.empty() || R.size() == cv::Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(D.total() == 4 && K.size() == cv::Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        cv::Matx33f camMat = K.getMat();
        f = cv::Vec2f(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        cv::Matx33d camMat = K.getMat();
        f = cv::Vec2d(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2d(camMat(0, 2), camMat(1, 2));
    }

    cv::Vec4d k = D.depth() == CV_32F ? (cv::Vec4d)*D.getMat().ptr<cv::Vec4f>(): *D.getMat().ptr<cv::Vec4d>();

    cv::Matx33d RR = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        RR = cv::Affine3d(rvec).rotation();
    }
    else if (!R.empty() && R.size() == cv::Size(3, 3))
        R.getMat().convertTo(RR, CV_64F);

    if(!P.empty())
    {
        cv::Matx33d PP;
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
        RR = PP * RR;
    }

    // start undistorting
    const cv::Vec2f* srcf = distorted.getMat().ptr<cv::Vec2f>();
    const cv::Vec2d* srcd = distorted.getMat().ptr<cv::Vec2d>();
    cv::Vec2f* dstf = undistorted.getMat().ptr<cv::Vec2f>();
    cv::Vec2d* dstd = undistorted.getMat().ptr<cv::Vec2d>();

    size_t n = distorted.total();
    int sdepth = distorted.depth();

    for(size_t i = 0; i < n; i++ )
    {
        cv::Vec2d pi = sdepth == CV_32F ? (cv::Vec2d)srcf[i] : srcd[i];  // image point
        cv::Vec2d pw((pi[0] - c[0])/f[0], (pi[1] - c[1])/f[1]);      // world point

        double scale = 1.0;

        double theta_d = sqrt(pw[0]*pw[0] + pw[1]*pw[1]);

        // the current camera model is only valid up to 180 FOV
        // for larger FOV the loop below does not converge
        // clip values so we still get plausible results for super fisheye images > 180 grad
        theta_d = std::min(std::max(-CV_PI/2., theta_d), CV_PI/2.);

        if (theta_d > 1e-8)
        {
            // compensate distortion iteratively
            double theta = theta_d;

            const double EPS = 1e-8; // or std::numeric_limits<double>::epsilon();
            for (int j = 0; j < 10; j++)
            {
                double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
                double k0_theta2 = k[0] * theta2, k1_theta4 = k[1] * theta4, k2_theta6 = k[2] * theta6, k3_theta8 = k[3] * theta8;
                /* new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta) *///牛顿迭代法求解多项式
                double theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                   (1 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8);
                theta = theta - theta_fix;
                if (fabs(theta_fix) < EPS)
                    break;
            }

			switch (mode)
			{
			case STEREOGRAPHIC:
				scale = 2 * std::tan(theta / 2) / theta_d;		//r = 2f * tan(theta / 2)
				break;
			case EQUIDISTANCE:
				scale = theta / theta_d;						// r = f * theta
				break;
			case EQUISOLID:
				scale = 2 * std::sin(theta / 2) / theta_d;		// r = 2f * sin(theta / 2)
				break;
			case ORTHOGONAL:
				scale = std::sin(theta) / theta_d;				// r = f * sin(theta)
				break;
			default:
				scale = std::tan(theta) / theta_d;				//r = f * tan(theta)
			}

        }

        cv::Vec2d pu = pw * scale; //undistorted point

        // reproject
        cv::Vec3d pr = RR * cv::Vec3d(pu[0], pu[1], 1.0); // rotated point optionally multiplied by new camera matrix
        cv::Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);       // final

        if( sdepth == CV_32F )
            dstf[i] = fi;
        else
            dstd[i] = fi;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::undistortPoints

void my_cv::fisheye::initUndistortRectifyMap( cv::InputArray K, cv::InputArray D, cv::InputArray R, cv::InputArray P,
    const cv::Size& size, int m1type, cv::OutputArray map1, cv::OutputArray map2, camMode mode)
{
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32F || m1type <=0 );
    map1.create( size, m1type <= 0 ? CV_16SC2 : m1type );
    map2.create( size, map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F );

    CV_Assert((K.depth() == CV_32F || K.depth() == CV_64F) && (D.depth() == CV_32F || D.depth() == CV_64F));
    CV_Assert((P.empty() || P.depth() == CV_32F || P.depth() == CV_64F) && (R.empty() || R.depth() == CV_32F || R.depth() == CV_64F));
    CV_Assert(K.size() == cv::Size(3, 3) && (D.empty() || D.total() == 4));
    CV_Assert(R.empty() || R.size() == cv::Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(P.empty() || P.size() == cv::Size(3, 3) || P.size() == cv::Size(4, 3));

    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        cv::Matx33f camMat = K.getMat();
        f = cv::Vec2f(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        cv::Matx33d camMat = K.getMat();
        f = cv::Vec2d(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2d(camMat(0, 2), camMat(1, 2));
    }

    cv::Vec4d k = cv::Vec4d::all(0);
    if (!D.empty())
        k = D.depth() == CV_32F ? (cv::Vec4d)*D.getMat().ptr<cv::Vec4f>(): *D.getMat().ptr<cv::Vec4d>();

    cv::Matx33d RR  = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        RR = cv::Affine3d(rvec).rotation();
    }
    else if (!R.empty() && R.size() == cv::Size(3, 3))
        R.getMat().convertTo(RR, CV_64F);

    cv::Matx33d PP = cv::Matx33d::eye();
    if (!P.empty())
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);

    cv::Matx33d iR = (PP * RR).inv(cv::DECOMP_SVD);

    for( int i = 0; i < size.height; ++i)
    {
        float* m1f = map1.getMat().ptr<float>(i);
        float* m2f = map2.getMat().ptr<float>(i);
        short*  m1 = (short*)m1f;
        ushort* m2 = (ushort*)m2f;

        double _x = i*iR(0, 1) + iR(0, 2),
               _y = i*iR(1, 1) + iR(1, 2),
               _w = i*iR(2, 1) + iR(2, 2);

        for( int j = 0; j < size.width; ++j)
        {
            double u, v;
            if( _w <= 0)
            {
                u = (_x > 0) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
                v = (_y > 0) ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
            }
            else
            {
                double x = _x/_w, y = _y/_w;

                double r = sqrt(x*x + y*y);

				double theta;
				switch (mode)
				{
				case STEREOGRAPHIC:
					theta = 2 * atan(r / 2);		//r = 2f * tan(theta / 2)
					break;
				case EQUIDISTANCE:
					theta = r;						// r = f * theta
					break;
				case EQUISOLID:
					theta = 2 * asin(r / 2);		// r = 2f * sin(theta / 2)
					break;
				case ORTHOGONAL:
					theta = asin(r);				// r = f * sin(theta)
					break;
				default:
					theta = atan(r);				//r = f * tan(theta)
				}

                double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta4*theta4;
                double theta_d = theta * (1 + k[0]*theta2 + k[1]*theta4 + k[2]*theta6 + k[3]*theta8);

                double scale = (r == 0) ? 1.0 : theta_d / r;
                u = f[0]*x*scale + c[0];
                v = f[1]*y*scale + c[1];
            }

            if( m1type == CV_16SC2 )
            {
                int iu = cv::saturate_cast<int>(u*cv::INTER_TAB_SIZE);
                int iv = cv::saturate_cast<int>(v*cv::INTER_TAB_SIZE);
                m1[j*2+0] = (short)(iu >> cv::INTER_BITS);
                m1[j*2+1] = (short)(iv >> cv::INTER_BITS);
                m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE-1)));
            }
            else if( m1type == CV_32FC1 )
            {
                m1f[j] = (float)u;
                m2f[j] = (float)v;
            }

            _x += iR(0, 0);
            _y += iR(1, 0);
            _w += iR(2, 0);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::undistortImage

void my_cv::fisheye::undistortImage(cv::InputArray distorted, cv::OutputArray undistorted,
        cv::InputArray K, cv::InputArray D, cv::InputArray Knew, const cv::Size& new_size)
{
	cv::Size size = !new_size.empty() ? new_size : distorted.size();

    cv::Mat map1, map2;
	double rote_theta_x, rote_theta_y;
	rote_theta_x = 0;//-PI / 12
	rote_theta_y = 0;//
	cv::Mat Ry, Rx, R;
	Ry = (cv::Mat_<double>(3, 3) << cos(rote_theta_y), 0, sin(rote_theta_y), 0, 1, 0, -sin(rote_theta_y), 0, cos(rote_theta_y));
	Rx = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cos(rote_theta_x), -sin(rote_theta_x), 0, sin(rote_theta_x), cos(rote_theta_x));
	R = Ry * Rx; //
	//R = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

    my_cv::fisheye::initUndistortRectifyMap(K, D, R, Knew, size, CV_16SC2, map1, map2, cur_fisheye_mode);
    cv::remap(distorted, undistorted, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::estimateNewCameraMatrixForUndistortRectify

void my_cv::fisheye::estimateNewCameraMatrixForUndistortRectify(cv::InputArray K, cv::InputArray D, const cv::Size &image_size, cv::InputArray R,
    cv::OutputArray P, double balance, const cv::Size& new_size, double fov_scale)
{
    CV_Assert( K.size() == cv::Size(3, 3)       && (K.depth() == CV_32F || K.depth() == CV_64F));
    CV_Assert(D.empty() || ((D.total() == 4) && (D.depth() == CV_32F || D.depth() == CV_64F)));

    int w = image_size.width, h = image_size.height;
    balance = std::min(std::max(balance, 0.0), 1.0);

    cv::Mat points(1, 4, CV_64FC2);
    cv::Vec2d* pptr = points.ptr<cv::Vec2d>();
    pptr[0] = cv::Vec2d(w/2, 0);
    pptr[1] = cv::Vec2d(w, h/2);
    pptr[2] = cv::Vec2d(w/2, h);
    pptr[3] = cv::Vec2d(0, h/2);

    my_cv::fisheye::undistortPoints(points, points, K, D, R, cv::noArray(), cur_fisheye_mode);
    cv::Scalar center_mass = mean(points);
    cv::Vec2d cn(center_mass.val);

    double aspect_ratio = (K.depth() == CV_32F) ? K.getMat().at<float >(0,0)/K.getMat().at<float> (1,1)
                                                : K.getMat().at<double>(0,0)/K.getMat().at<double>(1,1);

    // convert to identity ratio
    cn[0] *= aspect_ratio;
    for(size_t i = 0; i < points.total(); ++i)
        pptr[i][1] *= aspect_ratio;

    double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
    for(size_t i = 0; i < points.total(); ++i)
    {
        miny = std::min(miny, pptr[i][1]);
        maxy = std::max(maxy, pptr[i][1]);
        minx = std::min(minx, pptr[i][0]);
        maxx = std::max(maxx, pptr[i][0]);
    }

    double f1 = w * 0.5/(cn[0] - minx);
    double f2 = w * 0.5/(maxx - cn[0]);
    double f3 = h * 0.5 * aspect_ratio/(cn[1] - miny);
    double f4 = h * 0.5 * aspect_ratio/(maxy - cn[1]);

    double fmin = std::min(f1, std::min(f2, std::min(f3, f4)));
    double fmax = std::max(f1, std::max(f2, std::max(f3, f4)));

    double f = balance * fmin + (1.0 - balance) * fmax;
    f *= fov_scale > 0 ? 1.0/fov_scale : 1.0;

    cv::Vec2d new_f(f, f), new_c = -cn * f + cv::Vec2d(w, h * aspect_ratio) * 0.5;

    // restore aspect ratio
    new_f[1] /= aspect_ratio;
    new_c[1] /= aspect_ratio;

    if (!new_size.empty())
    {
        double rx = new_size.width /(double)image_size.width;
        double ry = new_size.height/(double)image_size.height;

        new_f[0] *= rx;  new_f[1] *= ry;
        new_c[0] *= rx;  new_c[1] *= ry;
    }

    cv::Mat(cv::Matx33d(new_f[0], 0, new_c[0],
                0, new_f[1], new_c[1],
                0,        0,       1)).convertTo(P, P.empty() ? K.type() : P.type());
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::stereoRectify

void my_cv::fisheye::stereoRectify( cv::InputArray K1, cv::InputArray D1, cv::InputArray K2, cv::InputArray D2, const cv::Size& imageSize,
        cv::InputArray _R, cv::InputArray _tvec, cv::OutputArray R1, cv::OutputArray R2, cv::OutputArray P1, cv::OutputArray P2,
        cv::OutputArray Q, int flags, const cv::Size& newImageSize, double balance, double fov_scale)
{

    CV_Assert((_R.size() == cv::Size(3, 3) || _R.total() * _R.channels() == 3) && (_R.depth() == CV_32F || _R.depth() == CV_64F));
    CV_Assert(_tvec.total() * _tvec.channels() == 3 && (_tvec.depth() == CV_32F || _tvec.depth() == CV_64F));


    cv::Mat aaa = _tvec.getMat().reshape(3, 1);

    cv::Vec3d rvec; // Rodrigues vector
    if (_R.size() == cv::Size(3, 3))
    {
        cv::Matx33d rmat;
        _R.getMat().convertTo(rmat, CV_64F);
        rvec = cv::Affine3d(rmat).rvec();
    }
    else if (_R.total() * _R.channels() == 3)
        _R.getMat().convertTo(rvec, CV_64F);

    cv::Vec3d tvec;
    _tvec.getMat().convertTo(tvec, CV_64F);

    // rectification algorithm
    rvec *= -0.5;              // get average rotation

    cv::Matx33d r_r;
    Rodrigues(rvec, r_r);  // rotate cameras to same orientation by averaging

    cv::Vec3d t = r_r * tvec;
    cv::Vec3d uu(t[0] > 0 ? 1 : -1, 0, 0);

    // calculate global Z rotation
    cv::Vec3d ww = t.cross(uu);
    double nw = norm(ww);
    if (nw > 0.0)
        ww *= acos(fabs(t[0])/cv::norm(t))/nw;

    cv::Matx33d wr;
    Rodrigues(ww, wr);

    // apply to both views
    cv::Matx33d ri1 = wr * r_r.t();
    cv::Mat(ri1, false).convertTo(R1, R1.empty() ? CV_64F : R1.type());
    cv::Matx33d ri2 = wr * r_r;
    cv::Mat(ri2, false).convertTo(R2, R2.empty() ? CV_64F : R2.type());
    cv::Vec3d tnew = ri2 * tvec;

    // calculate projection/camera matrices. these contain the relevant rectified image internal params (fx, fy=fx, cx, cy)
    cv::Matx33d newK1, newK2;
    estimateNewCameraMatrixForUndistortRectify(K1, D1, imageSize, R1, newK1, balance, newImageSize, fov_scale);
    estimateNewCameraMatrixForUndistortRectify(K2, D2, imageSize, R2, newK2, balance, newImageSize, fov_scale);

    double fc_new = std::min(newK1(1,1), newK2(1,1));
    cv::Point2d cc_new[2] = { cv::Vec2d(newK1(0, 2), newK1(1, 2)), cv::Vec2d(newK2(0, 2), newK2(1, 2)) };

    // Vertical focal length must be the same for both images to keep the epipolar constraint use fy for fx also.
    // For simplicity, set the principal points for both cameras to be the average
    // of the two principal points (either one of or both x- and y- coordinates)
    if( flags & cv::CALIB_ZERO_DISPARITY )
        cc_new[0] = cc_new[1] = (cc_new[0] + cc_new[1]) * 0.5;
    else
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;

    cv::Mat(cv::Matx34d(fc_new, 0, cc_new[0].x, 0,
                0, fc_new, cc_new[0].y, 0,
                0,      0,           1, 0), false).convertTo(P1, P1.empty() ? CV_64F : P1.type());

    cv::Mat(cv::Matx34d(fc_new, 0, cc_new[1].x, tnew[0]*fc_new, // baseline * focal length;,
                0, fc_new, cc_new[1].y,              0,
                0,      0,           1,              0), false).convertTo(P2, P2.empty() ? CV_64F : P2.type());

    if (Q.needed())
        cv::Mat(cv::Matx44d(1, 0, 0,           -cc_new[0].x,
                    0, 1, 0,           -cc_new[0].y,
                    0, 0, 0,            fc_new,
                    0, 0, -1./tnew[0], (cc_new[0].x - cc_new[1].x)/tnew[0]), false).convertTo(Q, Q.empty() ? CV_64F : Q.depth());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::calibrate

double my_cv::fisheye::calibrate(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints, const cv::Size& image_size,
                                    cv::InputOutputArray K, cv::InputOutputArray D, cv::OutputArrayOfArrays rvecs, cv::OutputArrayOfArrays tvecs,
                                    int flags , cv::TermCriteria criteria)
{

    CV_Assert(!objectPoints.empty() && !imagePoints.empty() && objectPoints.total() == imagePoints.total());
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
    CV_Assert(imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2);
    CV_Assert(K.empty() || (K.size() == cv::Size(3,3)));
    CV_Assert(D.empty() || (D.total() == 4));
    CV_Assert(rvecs.empty() || (rvecs.channels() == 3));
    CV_Assert(tvecs.empty() || (tvecs.channels() == 3));

    CV_Assert((!K.empty() && !D.empty()) || !(flags & cv::fisheye::CALIB_USE_INTRINSIC_GUESS));

    using namespace cv::internal;
    //-------------------------------Initialization
    my_cv::internal::IntrinsicParams finalParam;
	my_cv::internal::IntrinsicParams currentParam;
	my_cv::internal::IntrinsicParams errors;

    finalParam.isEstimate[0] = 1;
    finalParam.isEstimate[1] = 1;
    finalParam.isEstimate[2] = flags & cv::fisheye::CALIB_FIX_PRINCIPAL_POINT ? 0 : 1;
    finalParam.isEstimate[3] = flags & cv::fisheye::CALIB_FIX_PRINCIPAL_POINT ? 0 : 1;
    finalParam.isEstimate[4] = flags & cv::fisheye::CALIB_FIX_SKEW ? 0 : 1;
    finalParam.isEstimate[5] = flags & cv::fisheye::CALIB_FIX_K1 ? 0 : 1;
    finalParam.isEstimate[6] = flags & cv::fisheye::CALIB_FIX_K2 ? 0 : 1;
    finalParam.isEstimate[7] = flags & cv::fisheye::CALIB_FIX_K3 ? 0 : 1;
    finalParam.isEstimate[8] = flags & cv::fisheye::CALIB_FIX_K4 ? 0 : 1;

    const int recompute_extrinsic = flags & cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC ? 1: 0;
    const int check_cond = flags & cv::fisheye::CALIB_CHECK_COND ? 1 : 0;

    const double alpha_smooth = 0.4;
    const double thresh_cond = 1e6;
    double change = 1;
    cv::Vec2d err_std;

    cv::Matx33d _K;
    cv::Vec4d _D;
    if (flags & cv::fisheye::CALIB_USE_INTRINSIC_GUESS)
    {
        K.getMat().convertTo(_K, CV_64FC1);
        D.getMat().convertTo(_D, CV_64FC1);
        finalParam.Init(cv::Vec2d(_K(0,0), _K(1, 1)),
                        cv::Vec2d(_K(0,2), _K(1, 2)),
                        cv::Vec4d(flags & cv::fisheye::CALIB_FIX_K1 ? 0 : _D[0],
                              flags & cv::fisheye::CALIB_FIX_K2 ? 0 : _D[1],
                              flags & cv::fisheye::CALIB_FIX_K3 ? 0 : _D[2],
                              flags & cv::fisheye::CALIB_FIX_K4 ? 0 : _D[3]),
                        _K(0, 1) / _K(0, 0));
    }
    else
    {
        finalParam.Init(cv::Vec2d(std::max(image_size.width, image_size.height) / CV_PI, std::max(image_size.width, image_size.height) / CV_PI),
                        cv::Vec2d(image_size.width  / 2.0 - 0.5, image_size.height / 2.0 - 0.5));
    }

    errors.isEstimate = finalParam.isEstimate;

    std::vector<cv::Vec3d> omc(objectPoints.total()), Tc(objectPoints.total());

    CalibrateExtrinsics(objectPoints, imagePoints, finalParam, check_cond, thresh_cond, omc, Tc);


    //-------------------------------Optimization
    for(int iter = 0; iter < std::numeric_limits<int>::max(); ++iter)
    {
        if ((criteria.type == 1 && iter >= criteria.maxCount)  ||
            (criteria.type == 2 && change <= criteria.epsilon) ||
            (criteria.type == 3 && (change <= criteria.epsilon || iter >= criteria.maxCount)))
            break;

        double alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, iter + 1.0);

        cv::Mat JJ2, ex3;
        ComputeJacobians(objectPoints, imagePoints, finalParam, omc, Tc, check_cond,thresh_cond, JJ2, ex3);

        cv::Mat G;
        solve(JJ2, ex3, G);
        currentParam = finalParam + alpha_smooth2*G;

        change = norm(cv::Vec4d(currentParam.f[0], currentParam.f[1], currentParam.c[0], currentParam.c[1]) -
                cv::Vec4d(finalParam.f[0], finalParam.f[1], finalParam.c[0], finalParam.c[1]))
                / norm(cv::Vec4d(currentParam.f[0], currentParam.f[1], currentParam.c[0], currentParam.c[1]));

        finalParam = currentParam;

        if (recompute_extrinsic)
        {
            CalibrateExtrinsics(objectPoints,  imagePoints, finalParam, check_cond,
                                    thresh_cond, omc, Tc);
        }
    }

    //-------------------------------Validation
    double rms;
    EstimateUncertainties(objectPoints, imagePoints, finalParam,  omc, Tc, errors, err_std, thresh_cond,
                              check_cond, rms);

    //-------------------------------
    _K = cv::Matx33d(finalParam.f[0], finalParam.f[0] * finalParam.alpha, finalParam.c[0],
            0,                    finalParam.f[1], finalParam.c[1],
            0,                                  0,               1);

    if (K.needed()) cv::Mat(_K).convertTo(K, K.empty() ? CV_64FC1 : K.type());
    if (D.needed()) cv::Mat(finalParam.k).convertTo(D, D.empty() ? CV_64FC1 : D.type());
    if (rvecs.isMatVector())
    {
        int N = (int)objectPoints.total();

        if(rvecs.empty())
            rvecs.create(N, 1, CV_64FC3);

        if(tvecs.empty())
            tvecs.create(N, 1, CV_64FC3);

        for(int i = 0; i < N; i++ )
        {
            rvecs.create(3, 1, CV_64F, i, true);
            tvecs.create(3, 1, CV_64F, i, true);
            memcpy(rvecs.getMat(i).ptr(), omc[i].val, sizeof(cv::Vec3d));
            memcpy(tvecs.getMat(i).ptr(), Tc[i].val, sizeof(cv::Vec3d));
        }
    }
    else
    {
        if (rvecs.needed()) cv::Mat(omc).convertTo(rvecs, rvecs.empty() ? CV_64FC3 : rvecs.type());
        if (tvecs.needed()) cv::Mat(Tc).convertTo(tvecs, tvecs.empty() ? CV_64FC3 : tvecs.type());
    }

    return rms;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::stereoCalibrate

double my_cv::fisheye::stereoCalibrate(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints1, cv::InputArrayOfArrays imagePoints2,
                                    cv::InputOutputArray K1, cv::InputOutputArray D1, cv::InputOutputArray K2, cv::InputOutputArray D2, cv::Size imageSize,
                                    cv::OutputArray R, cv::OutputArray T, int flags, cv::TermCriteria criteria)
{

    CV_Assert(!objectPoints.empty() && !imagePoints1.empty() && !imagePoints2.empty());
    CV_Assert(objectPoints.total() == imagePoints1.total() || imagePoints1.total() == imagePoints2.total());
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
    CV_Assert(imagePoints1.type() == CV_32FC2 || imagePoints1.type() == CV_64FC2);
    CV_Assert(imagePoints2.type() == CV_32FC2 || imagePoints2.type() == CV_64FC2);

    CV_Assert(K1.empty() || (K1.size() == cv::Size(3,3)));
    CV_Assert(D1.empty() || (D1.total() == 4));
    CV_Assert(K2.empty() || (K1.size() == cv::Size(3,3)));
    CV_Assert(D2.empty() || (D1.total() == 4));

    CV_Assert((!K1.empty() && !K2.empty() && !D1.empty() && !D2.empty()) || !(flags & cv::fisheye::CALIB_FIX_INTRINSIC));

    //-------------------------------Initialization

    const int threshold = 5;
    const double thresh_cond = 1e6;
    const int check_cond = 1;

    int n_points = (int)objectPoints.getMat(0).total();
    int n_images = (int)objectPoints.total();

    double change = 1;

    my_cv::internal::IntrinsicParams intrinsicLeft;
	my_cv::internal::IntrinsicParams intrinsicRight;

	my_cv::internal::IntrinsicParams intrinsicLeft_errors;
	my_cv::internal::IntrinsicParams intrinsicRight_errors;

    cv::Matx33d _K1, _K2;
    cv::Vec4d _D1, _D2;
    if (!K1.empty()) K1.getMat().convertTo(_K1, CV_64FC1);
    if (!D1.empty()) D1.getMat().convertTo(_D1, CV_64FC1);
    if (!K2.empty()) K2.getMat().convertTo(_K2, CV_64FC1);
    if (!D2.empty()) D2.getMat().convertTo(_D2, CV_64FC1);

    std::vector<cv::Vec3d> rvecs1(n_images), tvecs1(n_images), rvecs2(n_images), tvecs2(n_images);

    if (!(flags & cv::fisheye::CALIB_FIX_INTRINSIC))
    {
        calibrate(objectPoints, imagePoints1, imageSize, _K1, _D1, rvecs1, tvecs1, flags, cv::TermCriteria(3, 20, 1e-6));
        calibrate(objectPoints, imagePoints2, imageSize, _K2, _D2, rvecs2, tvecs2, flags, cv::TermCriteria(3, 20, 1e-6));
    }

    intrinsicLeft.Init(cv::Vec2d(_K1(0,0), _K1(1, 1)), cv::Vec2d(_K1(0,2), _K1(1, 2)),
                       cv::Vec4d(_D1[0], _D1[1], _D1[2], _D1[3]), _K1(0, 1) / _K1(0, 0));

    intrinsicRight.Init(cv::Vec2d(_K2(0,0), _K2(1, 1)), cv::Vec2d(_K2(0,2), _K2(1, 2)),
                        cv::Vec4d(_D2[0], _D2[1], _D2[2], _D2[3]), _K2(0, 1) / _K2(0, 0));

    if ((flags & cv::fisheye::CALIB_FIX_INTRINSIC))
    {
        my_cv::internal::CalibrateExtrinsics(objectPoints,  imagePoints1, intrinsicLeft, check_cond, thresh_cond, rvecs1, tvecs1);
        my_cv::internal::CalibrateExtrinsics(objectPoints,  imagePoints2, intrinsicRight, check_cond, thresh_cond, rvecs2, tvecs2);
    }

    intrinsicLeft.isEstimate[0] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[1] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[2] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[3] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[4] = flags & (cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[5] = flags & (cv::fisheye::CALIB_FIX_K1 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[6] = flags & (cv::fisheye::CALIB_FIX_K2 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[7] = flags & (cv::fisheye::CALIB_FIX_K3 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[8] = flags & (cv::fisheye::CALIB_FIX_K4 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;

    intrinsicRight.isEstimate[0] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[1] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[2] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[3] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[4] = flags & (cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[5] = flags & (cv::fisheye::CALIB_FIX_K1 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[6] = flags & (cv::fisheye::CALIB_FIX_K2 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[7] = flags & (cv::fisheye::CALIB_FIX_K3 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[8] = flags & (cv::fisheye::CALIB_FIX_K4 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;

    intrinsicLeft_errors.isEstimate = intrinsicLeft.isEstimate;
    intrinsicRight_errors.isEstimate = intrinsicRight.isEstimate;

    std::vector<uchar> selectedParams;
    std::vector<uchar> tmp(6 * (n_images + 1), 1);
    selectedParams.insert(selectedParams.end(), intrinsicLeft.isEstimate.begin(), intrinsicLeft.isEstimate.end());
    selectedParams.insert(selectedParams.end(), intrinsicRight.isEstimate.begin(), intrinsicRight.isEstimate.end());
    selectedParams.insert(selectedParams.end(), tmp.begin(), tmp.end());

    //Init values for rotation and translation between two views
    cv::Mat om_list(1, n_images, CV_64FC3), T_list(1, n_images, CV_64FC3);
    cv::Mat om_ref, R_ref, T_ref, R1, R2;
    for (int image_idx = 0; image_idx < n_images; ++image_idx)
    {
        cv::Rodrigues(rvecs1[image_idx], R1);
        cv::Rodrigues(rvecs2[image_idx], R2);
        R_ref = R2 * R1.t();
        T_ref = cv::Mat(tvecs2[image_idx]) - R_ref * cv::Mat(tvecs1[image_idx]);
        cv::Rodrigues(R_ref, om_ref);
        om_ref.reshape(3, 1).copyTo(om_list.col(image_idx));
        T_ref.reshape(3, 1).copyTo(T_list.col(image_idx));
    }
    cv::Vec3d omcur = my_cv::internal::median3d(om_list);
    cv::Vec3d Tcur  = my_cv::internal::median3d(T_list);

    cv::Mat J = cv::Mat::zeros(4 * n_points * n_images, 18 + 6 * (n_images + 1), CV_64FC1),
            e = cv::Mat::zeros(4 * n_points * n_images, 1, CV_64FC1), Jkk, ekk;

    for(int iter = 0; ; ++iter)
    {
        if ((criteria.type == 1 && iter >= criteria.maxCount)  ||
            (criteria.type == 2 && change <= criteria.epsilon) ||
            (criteria.type == 3 && (change <= criteria.epsilon || iter >= criteria.maxCount)))
            break;

        J.create(4 * n_points * n_images, 18 + 6 * (n_images + 1), CV_64FC1);
        e.create(4 * n_points * n_images, 1, CV_64FC1);
        Jkk.create(4 * n_points, 18 + 6 * (n_images + 1), CV_64FC1);
        ekk.create(4 * n_points, 1, CV_64FC1);

        cv::Mat omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT;

        for (int image_idx = 0; image_idx < n_images; ++image_idx)
        {
            Jkk = cv::Mat::zeros(4 * n_points, 18 + 6 * (n_images + 1), CV_64FC1);

            cv::Mat object  = objectPoints.getMat(image_idx).clone();
            cv::Mat imageLeft  = imagePoints1.getMat(image_idx).clone();
            cv::Mat imageRight  = imagePoints2.getMat(image_idx).clone();
            cv::Mat jacobians, projected;

            //left camera jacobian
            cv::Mat rvec = cv::Mat(rvecs1[image_idx]);
            cv::Mat tvec  = cv::Mat(tvecs1[image_idx]);
            my_cv::internal::projectPoints(object, projected, rvec, tvec, intrinsicLeft, jacobians);
            cv::Mat(cv::Mat((imageLeft - projected).t()).reshape(1, 1).t()).copyTo(ekk.rowRange(0, 2 * n_points));
            jacobians.colRange(8, 11).copyTo(Jkk.colRange(24 + image_idx * 6, 27 + image_idx * 6).rowRange(0, 2 * n_points));
            jacobians.colRange(11, 14).copyTo(Jkk.colRange(27 + image_idx * 6, 30 + image_idx * 6).rowRange(0, 2 * n_points));
            jacobians.colRange(0, 2).copyTo(Jkk.colRange(0, 2).rowRange(0, 2 * n_points));
            jacobians.colRange(2, 4).copyTo(Jkk.colRange(2, 4).rowRange(0, 2 * n_points));
            jacobians.colRange(4, 8).copyTo(Jkk.colRange(5, 9).rowRange(0, 2 * n_points));
            jacobians.col(14).copyTo(Jkk.col(4).rowRange(0, 2 * n_points));

            //right camera jacobian
            my_cv::internal::compose_motion(rvec, tvec, omcur, Tcur, omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT);
            rvec = cv::Mat(rvecs2[image_idx]);
            tvec  = cv::Mat(tvecs2[image_idx]);

            my_cv::internal::projectPoints(object, projected, omr, Tr, intrinsicRight, jacobians);
            cv::Mat(cv::Mat((imageRight - projected).t()).reshape(1, 1).t()).copyTo(ekk.rowRange(2 * n_points, 4 * n_points));
            cv::Mat dxrdom = jacobians.colRange(8, 11) * domrdom + jacobians.colRange(11, 14) * dTrdom;
            cv::Mat dxrdT = jacobians.colRange(8, 11) * domrdT + jacobians.colRange(11, 14)* dTrdT;
            cv::Mat dxrdomckk = jacobians.colRange(8, 11) * domrdomckk + jacobians.colRange(11, 14) * dTrdomckk;
            cv::Mat dxrdTckk = jacobians.colRange(8, 11) * domrdTckk + jacobians.colRange(11, 14) * dTrdTckk;

            dxrdom.copyTo(Jkk.colRange(18, 21).rowRange(2 * n_points, 4 * n_points));
            dxrdT.copyTo(Jkk.colRange(21, 24).rowRange(2 * n_points, 4 * n_points));
            dxrdomckk.copyTo(Jkk.colRange(24 + image_idx * 6, 27 + image_idx * 6).rowRange(2 * n_points, 4 * n_points));
            dxrdTckk.copyTo(Jkk.colRange(27 + image_idx * 6, 30 + image_idx * 6).rowRange(2 * n_points, 4 * n_points));
            jacobians.colRange(0, 2).copyTo(Jkk.colRange(9 + 0, 9 + 2).rowRange(2 * n_points, 4 * n_points));
            jacobians.colRange(2, 4).copyTo(Jkk.colRange(9 + 2, 9 + 4).rowRange(2 * n_points, 4 * n_points));
            jacobians.colRange(4, 8).copyTo(Jkk.colRange(9 + 5, 9 + 9).rowRange(2 * n_points, 4 * n_points));
            jacobians.col(14).copyTo(Jkk.col(9 + 4).rowRange(2 * n_points, 4 * n_points));

            //check goodness of sterepair
            double abs_max  = 0;
            for (int i = 0; i < 4 * n_points; i++)
            {
                if (fabs(ekk.at<double>(i)) > abs_max)
                {
                    abs_max = fabs(ekk.at<double>(i));
                }
            }

            //CV_Assert(abs_max < threshold); // bad stereo pair
            if (abs_max < threshold)
                CV_Error(cv::Error::StsInternal, cv::format("bad stereo pair %d", image_idx));

            Jkk.copyTo(J.rowRange(image_idx * 4 * n_points, (image_idx + 1) * 4 * n_points));
            ekk.copyTo(e.rowRange(image_idx * 4 * n_points, (image_idx + 1) * 4 * n_points));
        }

        cv::Vec6d oldTom(Tcur[0], Tcur[1], Tcur[2], omcur[0], omcur[1], omcur[2]);

        //update all parameters
        my_cv::subMatrix(J, J, selectedParams, std::vector<uchar>(J.rows, 1));
        int a = cv::countNonZero(intrinsicLeft.isEstimate);
        int b = cv::countNonZero(intrinsicRight.isEstimate);
        cv::Mat deltas;
        solve(J.t() * J, J.t()*e, deltas);
        if (a > 0)
            intrinsicLeft = intrinsicLeft + deltas.rowRange(0, a);
        if (b > 0)
            intrinsicRight = intrinsicRight + deltas.rowRange(a, a + b);
        omcur = omcur + cv::Vec3d(deltas.rowRange(a + b, a + b + 3));
        Tcur = Tcur + cv::Vec3d(deltas.rowRange(a + b + 3, a + b + 6));
        for (int image_idx = 0; image_idx < n_images; ++image_idx)
        {
            rvecs1[image_idx] = cv::Mat(cv::Mat(rvecs1[image_idx]) + deltas.rowRange(a + b + 6 + image_idx * 6, a + b + 9 + image_idx * 6));
            tvecs1[image_idx] = cv::Mat(cv::Mat(tvecs1[image_idx]) + deltas.rowRange(a + b + 9 + image_idx * 6, a + b + 12 + image_idx * 6));
        }

        cv::Vec6d newTom(Tcur[0], Tcur[1], Tcur[2], omcur[0], omcur[1], omcur[2]);
        change = cv::norm(newTom - oldTom) / cv::norm(newTom);
    }

    double rms = 0;
    const cv::Vec2d* ptr_e = e.ptr<cv::Vec2d>();
    for (size_t i = 0; i < e.total() / 2; i++)
    {
        rms += ptr_e[i][0] * ptr_e[i][0] + ptr_e[i][1] * ptr_e[i][1];
    }

    rms /= ((double)e.total() / 2.0);
    rms = sqrt(rms);

    _K1 = cv::Matx33d(intrinsicLeft.f[0], intrinsicLeft.f[0] * intrinsicLeft.alpha, intrinsicLeft.c[0],
                                       0,                       intrinsicLeft.f[1], intrinsicLeft.c[1],
                                       0,                                        0,                 1);

    _K2 = cv::Matx33d(intrinsicRight.f[0], intrinsicRight.f[0] * intrinsicRight.alpha, intrinsicRight.c[0],
                                        0,                        intrinsicRight.f[1], intrinsicRight.c[1],
                                        0,                                          0,                  1);

    cv::Mat _R;
    Rodrigues(omcur, _R);

    if (K1.needed()) cv::Mat(_K1).convertTo(K1, K1.empty() ? CV_64FC1 : K1.type());
    if (K2.needed()) cv::Mat(_K2).convertTo(K2, K2.empty() ? CV_64FC1 : K2.type());
    if (D1.needed()) cv::Mat(intrinsicLeft.k).convertTo(D1, D1.empty() ? CV_64FC1 : D1.type());
    if (D2.needed()) cv::Mat(intrinsicRight.k).convertTo(D2, D2.empty() ? CV_64FC1 : D2.type());
    if (R.needed()) _R.convertTo(R, R.empty() ? CV_64FC1 : R.type());
    if (T.needed()) cv::Mat(Tcur).convertTo(T, T.empty() ? CV_64FC1 : T.type());

    return rms;
}

