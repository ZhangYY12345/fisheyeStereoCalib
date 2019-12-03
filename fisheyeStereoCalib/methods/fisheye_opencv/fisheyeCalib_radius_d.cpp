#include "fisheyeCalib_radius_d.h"
#include "fisheyeCalib_try.h"
#include "../polynomial-solve/root_finder.h"

extern camMode cur_fisheye_mode;

// fisheye raius distort
//

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::projectPoints

void my_cv::fisheye_r_d::projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints, const cv::Affine3d& affine,
	cv::InputArray K, cv::InputArray D, double alpha, cv::OutputArray jacobian, camMode mode)
{
	projectPoints(objectPoints, imagePoints, affine.rvec(), affine.translation(), K, D, alpha, jacobian, mode);
}

void my_cv::fisheye_r_d::projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints, cv::InputArray _rvec,
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
	cv::Vec3d T = _tvec.depth() == CV_32F ? (cv::Vec3d)*_tvec.getMat().ptr<cv::Vec3f>() : *_tvec.getMat().ptr<cv::Vec3d>();

	CV_Assert(_K.size() == cv::Size(3, 3) && (_K.type() == CV_32F || _K.type() == CV_64F) && _D.type() == _K.type() && _D.total() == 6);

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

	cv::Vec6d k = _D.depth() == CV_32F ? (cv::Vec6d)*_D.getMat().ptr<cv::Vec6f>() : *_D.getMat().ptr<cv::Vec6d>();

	const bool isJacobianNeeded = jacobian.needed();
	JacobianRow *Jn = 0;
	if (isJacobianNeeded)
	{
		int nvars = 2 + 2 + 1 + 6 + 3 + 3; // f, c, alpha, k, om, T,
		jacobian.create(2 * (int)n, nvars, CV_64F);
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

	for (size_t i = 0; i < n; ++i)
	{
		cv::Vec3d Xi = objectPoints.depth() == CV_32F ? (cv::Vec3d)Xf[i] : Xd[i];
		cv::Vec3d Y = aff * Xi;
		if (fabs(Y[2]) < DBL_MIN)
			Y[2] = 1;
		cv::Vec2d x(Y[0] / Y[2], Y[1] / Y[2]);

		double r_2 = x.dot(x);
		double r_ = std::sqrt(r_2);
		double theta = getTheta(r_, IDEAL_PERSPECTIVE);
		double r = getR(theta, mode);
		//double r = r_;

		// r_d = r(1 + k[0] * r^2 + k[1] * r^4 + k[2] * r^6 + k[3] * r^8)
		double r2 = r * r, r3 = r2 * r, r4 = r2 * r2, r5 = r4 * r,
			r6 = r3 * r3, r7 = r6 * r, r8 = r4 * r4, r9 = r8 * r, r10 = r8 * r2, r11 = r9 * r2, r12 = r10 * r2, r13 = r11 * r2;

		double r_d = r + k[0] * r3 + k[1] * r5 + k[2] * r7 + k[3] * r9 + k[4] * r11 + k[5] * r13;

		double inv_r_ = r_ > 1e-8 ? 1.0 / r_ : 1;
		double cdist = r_ > 1e-8 ? r_d * inv_r_ : 1;

		cv::Vec2d xd1 = x * cdist;
		cv::Vec2d xd3(xd1[0] + alpha * xd1[1], xd1[1]);
		cv::Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);

		if (objectPoints.depth() == CV_32F)
			xpf[i] = final_point;
		else
			xpd[i] = final_point;

		if (isJacobianNeeded)
		{
			//cv::Vec3d Xi = pdepth == CV_32F ? (cv::Vec3d)Xf[i] : Xd[i];
			//cv::Vec3d Y = aff*Xi;				Y---->相机坐标系
			double dYdR[] = { Xi[0], Xi[1], Xi[2], 0, 0, 0, 0, 0, 0,
							  0, 0, 0, Xi[0], Xi[1], Xi[2], 0, 0, 0,
							  0, 0, 0, 0, 0, 0, Xi[0], Xi[1], Xi[2] };//?todo

			cv::Matx33d dYdom_data = cv::Matx<double, 3, 9>(dYdR) * dRdom.t();
			const cv::Vec3d *dYdom = (cv::Vec3d*)dYdom_data.val;

			cv::Matx33d dYdT_data = cv::Matx33d::eye();
			const cv::Vec3d *dYdT = (cv::Vec3d*)dYdT_data.val;

			//cv::Vec2d x(Y[0]/Y[2], Y[1]/Y[2]);	x--->归一化相机坐标系
			cv::Vec3d dxdom[2];
			dxdom[0] = (1.0 / Y[2]) * dYdom[0] - x[0] / Y[2] * dYdom[2];
			dxdom[1] = (1.0 / Y[2]) * dYdom[1] - x[1] / Y[2] * dYdom[2];

			cv::Vec3d dxdT[2];
			dxdT[0] = (1.0 / Y[2]) * dYdT[0] - x[0] / Y[2] * dYdT[2];
			dxdT[1] = (1.0 / Y[2]) * dYdT[1] - x[1] / Y[2] * dYdT[2];

			//double r_2 = x.dot(x);
			cv::Vec3d dr_2dom = 2 * x[0] * dxdom[0] + 2 * x[1] * dxdom[1];
			cv::Vec3d dr_2dT = 2 * x[0] * dxdT[0] + 2 * x[1] * dxdT[1];

			//double r_ = std::sqrt(r_2);
			double dr_dr_2 = r_ > 1e-8 ? 1.0/(2.0*r_) : 1;
			cv::Vec3d dr_dom = dr_dr_2 * dr_2dom;
			cv::Vec3d dr_dT = dr_dr_2 * dr_2dT;

			double dthetadr_ = get_dthetadr(r_, IDEAL_PERSPECTIVE);
			cv::Vec3d dthetadom = dthetadr_ * dr_dom;
			cv::Vec3d dthetadT = dthetadr_ * dr_dT;
			double drdtheta = get_drdtheta(theta, mode);
			cv::Vec3d drdom = drdtheta * dthetadom;
			cv::Vec3d drdT = drdtheta * dthetadT;

			//cv::Vec3d drdom = dr_dom;
			//cv::Vec3d drdT = dr_dT;

			//double r_d = r + k[0]*r3 + k[1]*r5 + k[2]*r7 + k[3]*r9;
			double dr_ddr = 1 + 3 * k[0] * r2 + 5 * k[1] * r4 + 7 * k[2] * r6 + 9 * k[3] * r8 + 11 * k[4] * r10 + 13 * k[5] * r12;
			cv::Vec3d dr_ddom = dr_ddr * drdom;
			cv::Vec3d dr_ddT = dr_ddr * drdT;
			cv::Vec6d dr_ddk = cv::Vec6d(r3, r5, r7, r9, r11, r13);

			//double inv_r_ = r_ > 1e-8 ? 1.0/r_ : 1;
			//double cdist = r_ > 1e-8 ? r_d / r_ : 1;

			cv::Vec3d dcdistdom = inv_r_ * (dr_ddom - cdist * dr_dom);
			cv::Vec3d dcdistdT = inv_r_ * (dr_ddT - cdist * dr_dT);
			cv::Vec6d dcdistdk = inv_r_ * dr_ddk;

			//cv::Vec2d xd1 = x * cdist;
			cv::Vec6d dxd1dk[2];
			cv::Vec3d dxd1dom[2], dxd1dT[2];
			dxd1dom[0] = x[0] * dcdistdom + cdist * dxdom[0];
			dxd1dom[1] = x[1] * dcdistdom + cdist * dxdom[1];
			dxd1dT[0] = x[0] * dcdistdT + cdist * dxdT[0];
			dxd1dT[1] = x[1] * dcdistdT + cdist * dxdT[1];
			dxd1dk[0] = x[0] * dcdistdk;
			dxd1dk[1] = x[1] * dcdistdk;

			//cv::Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
			cv::Vec6d dxd3dk[2];
			cv::Vec3d dxd3dom[2], dxd3dT[2];
			dxd3dom[0] = dxd1dom[0] + alpha * dxd1dom[1];
			dxd3dom[1] = dxd1dom[1];
			dxd3dT[0] = dxd1dT[0] + alpha * dxd1dT[1];
			dxd3dT[1] = dxd1dT[1];
			dxd3dk[0] = dxd1dk[0] + alpha * dxd1dk[1];
			dxd3dk[1] = dxd1dk[1];

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

void my_cv::fisheye_r_d::distortPoints(cv::InputArray undistorted, cv::OutputArray distorted, cv::InputArray K, cv::InputArray D, double alpha, camMode mode)
{
	// will support only 2-channel data now for points
	CV_Assert(undistorted.type() == CV_32FC2 || undistorted.type() == CV_64FC2);
	distorted.create(undistorted.size(), undistorted.type());
	size_t n = undistorted.total();

	CV_Assert(K.size() == cv::Size(3, 3) && (K.type() == CV_32F || K.type() == CV_64F) && D.total() == 6);

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

	cv::Vec6d k = D.depth() == CV_32F ? (cv::Vec6d)*D.getMat().ptr<cv::Vec6f>() : *D.getMat().ptr<cv::Vec6d>();

	const cv::Vec2f* Xf = undistorted.getMat().ptr<cv::Vec2f>();
	const cv::Vec2d* Xd = undistorted.getMat().ptr<cv::Vec2d>();
	cv::Vec2f *xpf = distorted.getMat().ptr<cv::Vec2f>();
	cv::Vec2d *xpd = distorted.getMat().ptr<cv::Vec2d>();

	for (size_t i = 0; i < n; ++i)
	{
		cv::Vec2d x = undistorted.depth() == CV_32F ? (cv::Vec2d)Xf[i] : Xd[i];//理想图像像素坐标系
		cv::Vec2d px = cv::Vec2d((x[0] - c[0]) / f[0], (x[1] - c[1]) / f[1]);
		px[0] -= alpha * px[1];

		double r_2 = px.dot(px);
		double r = std::sqrt(r_2);

		double r2 = r * r, r3 = r2 * r, r5 = r3 * r2, r7 = r5 * r2, r9 = r7 * r2, r11 = r9 * r2, r13 = r11 * r2;

		double r_d = r + k[0] * r3 + k[1] * r5 + k[2] * r7 + k[3] * r9 + k[4] * r11 + k[5] * r13;

		double inv_r = r > 1e-8 ? 1.0 / r : 1;
		double cdist = r > 1e-8 ? r_d * inv_r : 1;

		cv::Vec2d xd1 = x * cdist;
		cv::Vec2d xd3(xd1[0] + alpha * xd1[1], xd1[1]);
		cv::Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);

		if (undistorted.depth() == CV_32F)
			xpf[i] = final_point;
		else
			xpd[i] = final_point;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::undistortPoints

void my_cv::fisheye_r_d::undistortPoints(cv::InputArray distorted, cv::OutputArray undistorted,
	cv::InputArray K, cv::InputArray D, cv::InputArray R, cv::InputArray P, camMode mode)
{
	// will support only 2-channel data now for points
	CV_Assert(distorted.type() == CV_32FC2 || distorted.type() == CV_64FC2);
	undistorted.create(distorted.size(), distorted.type());

	CV_Assert(P.empty() || P.size() == cv::Size(3, 3) || P.size() == cv::Size(4, 3));
	CV_Assert(R.empty() || R.size() == cv::Size(3, 3) || R.total() * R.channels() == 3);
	CV_Assert(D.total() == 6 && K.size() == cv::Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

	cv::Vec2d f, c;
	double alpha;
	if (K.depth() == CV_32F)
	{
		cv::Matx33f camMat = K.getMat();
		f = cv::Vec2f(camMat(0, 0), camMat(1, 1));
		c = cv::Vec2f(camMat(0, 2), camMat(1, 2));
		alpha = camMat(0, 1) / camMat(0, 0);
	}
	else
	{
		cv::Matx33d camMat = K.getMat();
		f = cv::Vec2d(camMat(0, 0), camMat(1, 1));
		c = cv::Vec2d(camMat(0, 2), camMat(1, 2));
		alpha = camMat(0, 1) / camMat(0, 0);
	}

	cv::Vec6d k = D.depth() == CV_32F ? (cv::Vec6d)*D.getMat().ptr<cv::Vec6f>() : *D.getMat().ptr<cv::Vec6d>();

	cv::Matx33d RR = cv::Matx33d::eye();
	if (!R.empty() && R.total() * R.channels() == 3)
	{
		cv::Vec3d rvec;
		R.getMat().convertTo(rvec, CV_64F);
		RR = cv::Affine3d(rvec).rotation();
	}
	else if (!R.empty() && R.size() == cv::Size(3, 3))
		R.getMat().convertTo(RR, CV_64F);

	cv::Matx33d PP = cv::Matx33d::eye();
	cv::Matx33d RRR = cv::Matx33d::eye();
	if (!P.empty())
	{
		P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
		RRR = PP * RR;//PP对应内参矩阵， 
	}

	// start undistorting
	const cv::Vec2f* srcf = distorted.getMat().ptr<cv::Vec2f>();
	const cv::Vec2d* srcd = distorted.getMat().ptr<cv::Vec2d>();
	cv::Vec2f* dstf = undistorted.getMat().ptr<cv::Vec2f>();
	cv::Vec2d* dstd = undistorted.getMat().ptr<cv::Vec2d>();

	size_t n = distorted.total();
	int sdepth = distorted.depth();

	for (size_t i = 0; i < n; i++)
	{
		cv::Vec2d pi = sdepth == CV_32F ? (cv::Vec2d)srcf[i] : srcd[i];  // image point
		cv::Vec2d pw((pi[0] - c[0]) / f[0], (pi[1] - c[1]) / f[1]);      // 
		pw[0] -= alpha * pw[1];

		double scale = 1.0;

		double r_d = sqrt(pw[0] * pw[0] + pw[1] * pw[1]);

		double r = r_d;
		if (r_d > 1e-8)
		{
			// compensate distortion iteratively
			//r = r_d;

			Eigen::VectorXd coeffs(14);
			coeffs(13) = -r_d;
			coeffs(12) = 1;
			coeffs(11) = coeffs(9) = coeffs(7) = coeffs(5) = coeffs(3) = coeffs(1) = 0;
			coeffs(10) = k[0];
			coeffs(8) = k[1];
			coeffs(6) = k[2];
			coeffs(4) = k[3];
			coeffs(2) = k[4];
			coeffs(0) = k[5];

			std::set<double> r_s;
			r_s = RootFinder::solvePolyInterval(coeffs, -INFINITY, INFINITY, 1e-8, false);
			double diff_r = INFINITY;
			for(std::set<double>::iterator r_si = r_s.begin(); r_si != r_s.end(); r_si++)
			{
				double cur_diff = fabs(r_d - *r_si);
				if(cur_diff < diff_r)
				{
					diff_r = cur_diff;
					r = *r_si;
				}
			}

			scale = r / r_d;
		}

		cv::Vec2d pu = pw * scale; //undistorted point in the image space

		// reproject
		double theta = getTheta(r, mode);

		cv::Vec2d pfi = pw / r_d;
		cv::Vec3d pr3d = cv::Vec3d(sin(theta) * pfi[0], sin(theta) * pfi[1], cos(theta));
		cv::Vec3d rotate_pr3d = RR * pr3d;
		if(rotate_pr3d[2] < DBL_MIN)
		{
			rotate_pr3d[2] = 1.0;
		}
		cv::Vec2d new_Xc = cv::Vec2d(rotate_pr3d[0] / rotate_pr3d[2], rotate_pr3d[1] / rotate_pr3d[2]);
		double new_r_2 = new_Xc.dot(new_Xc);
		double new_r_ = sqrt(new_r_2);
		double new_theta = getTheta(new_r_, IDEAL_PERSPECTIVE);
		double new_r = getR(new_theta, mode);
		cv::Vec2d new_pu = new_r * pfi;
		cv::Vec2d xd3(new_pu[0] + alpha * new_pu[1], new_pu[1]);
		cv::Vec2d fi(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);

		if (sdepth == CV_32F)
			dstf[i] = fi;
		else
			dstd[i] = fi;
	}
}

/**
 * \brief used in my_cv::internal::NormalizePixels(), to output the corresponding  camera coordination
 * \param distorted 
 * \param undistorted 
 * \param K 
 * \param D 
 * \param mode 
 */
void my_cv::fisheye_r_d::undistortPoints_H(cv::InputArray distorted, cv::OutputArray undistorted, cv::InputArray K,
	cv::InputArray D, camMode mode)
{
	// will support only 2-channel data now for points
	CV_Assert(distorted.type() == CV_32FC2 || distorted.type() == CV_64FC2);
	undistorted.create(distorted.size(), distorted.type());
	//if(distorted.type() == CV_32FC2)
	//{
	//	undistorted.create(distorted.size(), CV_32FC3);
	//}
	//else if(distorted.type() == CV_64FC2)
	//{
	//	undistorted.create(distorted.size(), CV_64FC3);
	//}
	

	CV_Assert(D.total() == 6 && K.size() == cv::Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

	cv::Vec2d f, c;
	double alpha;
	if (K.depth() == CV_32F)
	{
		cv::Matx33f camMat = K.getMat();
		f = cv::Vec2f(camMat(0, 0), camMat(1, 1));
		c = cv::Vec2f(camMat(0, 2), camMat(1, 2));
		alpha = camMat(0, 1) / camMat(0, 0);
	}
	else
	{
		cv::Matx33d camMat = K.getMat();
		f = cv::Vec2d(camMat(0, 0), camMat(1, 1));
		c = cv::Vec2d(camMat(0, 2), camMat(1, 2));
		alpha = camMat(0, 1) / camMat(0, 0);
	}

	cv::Vec6d k = D.depth() == CV_32F ? (cv::Vec6d)*D.getMat().ptr<cv::Vec6f>() : *D.getMat().ptr<cv::Vec6d>();

	// start undistorting
	const cv::Vec2f* srcf = distorted.getMat().ptr<cv::Vec2f>();
	const cv::Vec2d* srcd = distorted.getMat().ptr<cv::Vec2d>();
	cv::Vec2f* dstf = undistorted.getMat().ptr<cv::Vec2f>();
	cv::Vec2d* dstd = undistorted.getMat().ptr<cv::Vec2d>();

	size_t n = distorted.total();
	int sdepth = distorted.depth();

	for (size_t i = 0; i < n; i++)
	{
		cv::Vec2d pi = sdepth == CV_32F ? (cv::Vec2d)srcf[i] : srcd[i];  // image point
		cv::Vec2d pw((pi[0] - c[0]) / f[0], (pi[1] - c[1]) / f[1]);      // 

		double scale = 1.0;

		double r_d = sqrt(pw[0] * pw[0] + pw[1] * pw[1]);

		double r = r_d;
		if (r_d > 1e-8)
		{
			Eigen::VectorXd coeffs(14);
			coeffs(13) = -r_d;
			coeffs(12) = 1;
			coeffs(11) = coeffs(9) = coeffs(7) = coeffs(5) = coeffs(3) = coeffs(1) = 0;
			coeffs(10) = k[0];
			coeffs(8) = k[1];
			coeffs(6) = k[2];
			coeffs(4) = k[3];
			coeffs(2) = k[4];
			coeffs(0) = k[5];

			std::set<double> r_s;
			r_s = RootFinder::solvePolyInterval(coeffs, -INFINITY, INFINITY, 1e-7, false);
			double diff_r = INFINITY;
			for (std::set<double>::iterator r_si = r_s.begin(); r_si != r_s.end(); r_si++)
			{
				if(*r_si < 1e-8)
				{
					continue;
				}
				double cur_diff = fabs(r_d - *r_si);
				if (cur_diff < diff_r)
				{
					diff_r = cur_diff;
					r = *r_si;
				}
			}
		}

		//cv::Vec2d pu = pw / r_d; //undistorted point in the image space
		//if (sdepth == CV_32F)
		//	dstf[i] = pu;
		//else
		//	dstd[i] = pu;

		// reproject
		double theta = getTheta(r, mode);
		double r_ = getR(theta, IDEAL_PERSPECTIVE);
		double newScale = r_ / r_d;
		cv::Vec2d pfi = pw * newScale;//理想归一化相机坐标系
		//cv::Vec3d fi = cv::Vec3d(pfi[0], pfi[1], cos(theta));

		if (sdepth == CV_32F)
			dstf[i] = pfi;
		else
			dstd[i] = pfi;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::undistortPoints

void my_cv::fisheye_r_d::initUndistortRectifyMap(cv::InputArray K, cv::InputArray D, cv::InputArray R, cv::InputArray P,
	const cv::Size& size, int m1type, cv::OutputArray map1, cv::OutputArray map2, camMode mode)
{
	CV_Assert(m1type == CV_16SC2 || m1type == CV_32F || m1type <= 0);
	map1.create(size, m1type <= 0 ? CV_16SC2 : m1type);
	map2.create(size, map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F);

	CV_Assert((K.depth() == CV_32F || K.depth() == CV_64F) && (D.depth() == CV_32F || D.depth() == CV_64F));
	CV_Assert((P.empty() || P.depth() == CV_32F || P.depth() == CV_64F) && (R.empty() || R.depth() == CV_32F || R.depth() == CV_64F));
	CV_Assert(K.size() == cv::Size(3, 3) && (D.empty() || D.total() == 6));
	CV_Assert(R.empty() || R.size() == cv::Size(3, 3) || R.total() * R.channels() == 3);
	CV_Assert(P.empty() || P.size() == cv::Size(3, 3) || P.size() == cv::Size(4, 3));

	cv::Vec2d f, c;
	double alpha;
	if (K.depth() == CV_32F)
	{
		cv::Matx33f camMat = K.getMat();
		f = cv::Vec2f(camMat(0, 0), camMat(1, 1));
		c = cv::Vec2f(camMat(0, 2), camMat(1, 2));
		alpha = camMat(0, 1) / camMat(0, 0);
	}
	else
	{
		cv::Matx33d camMat = K.getMat();
		f = cv::Vec2d(camMat(0, 0), camMat(1, 1));
		c = cv::Vec2d(camMat(0, 2), camMat(1, 2));
		alpha = camMat(0, 1) / camMat(0, 0);
	}

	cv::Vec6d k = cv::Vec6d::all(0);
	if (!D.empty())
		k = D.depth() == CV_32F ? (cv::Vec6d)*D.getMat().ptr<cv::Vec6f>() : *D.getMat().ptr<cv::Vec6d>();

	cv::Matx33d RR = cv::Matx33d::eye();
	if (!R.empty() && R.total() * R.channels() == 3)
	{
		cv::Vec3d rvec;
		R.getMat().convertTo(rvec, CV_64F);
		RR = cv::Affine3d(rvec).rotation();
	}
	else if (!R.empty() && R.size() == cv::Size(3, 3))
		R.getMat().convertTo(RR, CV_64F);

	cv::Matx33d PP = cv::Matx33d::eye();
	cv::Vec2d p_f, p_c;
	double p_alpha;
	if (!P.empty())
	{
		P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
		p_f = cv::Vec2d(PP(0, 0), PP(1, 1));
		p_c = cv::Vec2d(PP(0, 2), PP(1, 2));
		p_alpha = PP(0, 1) / PP(0, 0);
	}
	else
	{
		p_f = f;
		p_c = c;
		p_alpha = alpha;
	}

	cv::Matx33d iR = RR.inv(cv::DECOMP_SVD);

	for (int i = 0; i < size.height; ++i)
	{
		float* m1f = map1.getMat().ptr<float>(i);
		float* m2f = map2.getMat().ptr<float>(i);
		short*  m1 = (short*)m1f;
		ushort* m2 = (ushort*)m2f;

		for (int j = 0; j < size.width; ++j)
		{
			cv::Vec2d x(j, i); //( u_ideal, v_ideal )
			cv::Vec2d px = cv::Vec2d((x[0] - p_c[0]) / p_f[0], (x[1] - p_c[1]) / p_f[1]);
			px[0] -= p_alpha * px[1];

			double r2 = px.dot(px);
			double r = std::sqrt(r2);
			cv::Vec2d pu = px / (1.0 * r);
			double theta = getTheta(r, mode);

			// 旋转后的相机坐标系坐标
			cv::Vec3d pr = iR * cv::Vec3d(sin(theta) * pu[0], sin(theta) * pu[1], cos(theta)); // rotated point optionally multiplied by new camera matrix
			if (pr[2] < DBL_MIN)
			{
				pr[2] = 1.0;
			}
			cv::Vec2d new_Xc = cv::Vec2d(pr[0] / pr[2], pr[1] / pr[2]);
			double new_r_2 = new_Xc.dot(new_Xc);
			double new_r_ = sqrt(new_r_2);
			double new_theta = getTheta(new_r_, IDEAL_PERSPECTIVE);
			double new_r = getR(new_theta, mode);

			double new_r2 = new_r * new_r, new_r4 = new_r2 * new_r2, new_r6 = new_r4 * new_r2, new_r8 = new_r4 * new_r4, new_r10 = new_r2 * new_r8, new_r12 = new_r2 * new_r10;
			double r_d = new_r * (1 + k[0] * new_r2 + k[1] * new_r4 + k[2] * new_r6 + k[3] * new_r8 + k[4] * new_r10 + k[5] * new_r12);

			double inv_r = new_r_ > 1e-8 ? 1.0 / new_r_ : 1;
			double cdist = new_r_ > 1e-8 ? r_d * inv_r : 1;

			cv::Vec2d xd1 = new_Xc * cdist;
			cv::Vec2d xd3(xd1[0] + alpha * xd1[1], xd1[1]);
			cv::Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);


			if (m1type == CV_16SC2)
			{
				int iu = cv::saturate_cast<int>(final_point[0]*cv::INTER_TAB_SIZE);
				int iv = cv::saturate_cast<int>(final_point[1]*cv::INTER_TAB_SIZE);
				m1[j * 2 + 0] = (short)(iu >> cv::INTER_BITS);
				m1[j * 2 + 1] = (short)(iv >> cv::INTER_BITS);
				m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE - 1))*cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE - 1)));
			}
			else if (m1type == CV_32FC1)
			{
				m1f[j] = (float)final_point[0];
				m2f[j] = (float)final_point[1];
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::undistortImage

void my_cv::fisheye_r_d::undistortImage(cv::InputArray distorted, cv::OutputArray undistorted,
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

	my_cv::fisheye_r_d::initUndistortRectifyMap(K, D, R, Knew, size, CV_16SC2, map1, map2, cur_fisheye_mode);
	cv::remap(distorted, undistorted, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::estimateNewCameraMatrixForUndistortRectify

void my_cv::fisheye_r_d::estimateNewCameraMatrixForUndistortRectify(cv::InputArray K, cv::InputArray D, const cv::Size &image_size, cv::InputArray R,
	cv::OutputArray P, double balance, const cv::Size& new_size, double fov_scale)
{
	CV_Assert(K.size() == cv::Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));
	CV_Assert(D.empty() || ((D.total() == 6) && (D.depth() == CV_32F || D.depth() == CV_64F)));

	int w = image_size.width, h = image_size.height;
	balance = std::min(std::max(balance, 0.0), 1.0);

	cv::Mat points(1, 4, CV_64FC2);
	cv::Vec2d* pptr = points.ptr<cv::Vec2d>();
	pptr[0] = cv::Vec2d(w / 2, 0);
	pptr[1] = cv::Vec2d(w, h / 2);
	pptr[2] = cv::Vec2d(w / 2, h);
	pptr[3] = cv::Vec2d(0, h / 2);

	my_cv::fisheye_r_d::undistortPoints(points, points, K, D, R, cv::noArray(), cur_fisheye_mode);
	cv::Scalar center_mass = mean(points);
	cv::Vec2d cn(center_mass.val);

	double aspect_ratio = (K.depth() == CV_32F) ? K.getMat().at<float >(0, 0) / K.getMat().at<float>(1, 1)
		: K.getMat().at<double>(0, 0) / K.getMat().at<double>(1, 1);

	// convert to identity ratio
	cn[0] *= aspect_ratio;
	for (size_t i = 0; i < points.total(); ++i)
		pptr[i][1] *= aspect_ratio;

	double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
	for (size_t i = 0; i < points.total(); ++i)
	{
		miny = std::min(miny, pptr[i][1]);
		maxy = std::max(maxy, pptr[i][1]);
		minx = std::min(minx, pptr[i][0]);
		maxx = std::max(maxx, pptr[i][0]);
	}

	double f1 = w * 0.5 / (cn[0] - minx);
	double f2 = w * 0.5 / (maxx - cn[0]);
	double f3 = h * 0.5 * aspect_ratio / (cn[1] - miny);
	double f4 = h * 0.5 * aspect_ratio / (maxy - cn[1]);

	double fmin = std::min(f1, std::min(f2, std::min(f3, f4)));
	double fmax = std::max(f1, std::max(f2, std::max(f3, f4)));

	double f = balance * fmin + (1.0 - balance) * fmax;
	f *= fov_scale > 0 ? 1.0 / fov_scale : 1.0;

	cv::Vec2d new_f(f, f), new_c = -cn * f + cv::Vec2d(w, h * aspect_ratio) * 0.5;

	// restore aspect ratio
	new_f[1] /= aspect_ratio;
	new_c[1] /= aspect_ratio;

	if (!new_size.empty())
	{
		double rx = new_size.width / (double)image_size.width;
		double ry = new_size.height / (double)image_size.height;

		new_f[0] *= rx;  new_f[1] *= ry;
		new_c[0] *= rx;  new_c[1] *= ry;
	}

	cv::Mat(cv::Matx33d(new_f[0], 0, new_c[0],
		0, new_f[1], new_c[1],
		0, 0, 1)).convertTo(P, P.empty() ? K.type() : P.type());
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::stereoRectify

void my_cv::fisheye_r_d::stereoRectify(cv::InputArray K1, cv::InputArray D1, cv::InputArray K2, cv::InputArray D2, const cv::Size& imageSize,
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
		ww *= acos(fabs(t[0]) / cv::norm(t)) / nw;

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

	double fc_new = std::min(newK1(1, 1), newK2(1, 1));
	cv::Point2d cc_new[2] = { cv::Vec2d(newK1(0, 2), newK1(1, 2)), cv::Vec2d(newK2(0, 2), newK2(1, 2)) };

	// Vertical focal length must be the same for both images to keep the epipolar constraint use fy for fx also.
	// For simplicity, set the principal points for both cameras to be the average
	// of the two principal points (either one of or both x- and y- coordinates)
	if (flags & cv::CALIB_ZERO_DISPARITY)
		cc_new[0] = cc_new[1] = (cc_new[0] + cc_new[1]) * 0.5;
	else
		cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;

	cv::Mat(cv::Matx34d(fc_new, 0, cc_new[0].x, 0,
		0, fc_new, cc_new[0].y, 0,
		0, 0, 1, 0), false).convertTo(P1, P1.empty() ? CV_64F : P1.type());

	cv::Mat(cv::Matx34d(fc_new, 0, cc_new[1].x, tnew[0] * fc_new, // baseline * focal length;,
		0, fc_new, cc_new[1].y, 0,
		0, 0, 1, 0), false).convertTo(P2, P2.empty() ? CV_64F : P2.type());

	if (Q.needed())
		cv::Mat(cv::Matx44d(1, 0, 0, -cc_new[0].x,
			0, 1, 0, -cc_new[0].y,
			0, 0, 0, fc_new,
			0, 0, -1. / tnew[0], (cc_new[0].x - cc_new[1].x) / tnew[0]), false).convertTo(Q, Q.empty() ? CV_64F : Q.depth());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::calibrate

double my_cv::fisheye_r_d::calibrate(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints, const cv::Size& image_size,
	cv::InputOutputArray K, cv::InputOutputArray D, cv::OutputArrayOfArrays rvecs, cv::OutputArrayOfArrays tvecs,
	int flags, cv::TermCriteria criteria)
{

	CV_Assert(!objectPoints.empty() && !imagePoints.empty() && objectPoints.total() == imagePoints.total());
	CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
	CV_Assert(imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2);
	CV_Assert(K.empty() || (K.size() == cv::Size(3, 3)));
	CV_Assert(D.empty() || (D.total() == 6));
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
	finalParam.isEstimate[9] = 1;
	finalParam.isEstimate[10] = 1;

	const int recompute_extrinsic = flags & cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC ? 1 : 0;
	const int check_cond = flags & cv::fisheye::CALIB_CHECK_COND ? 1 : 0;

	const double alpha_smooth = 0.4;
	const double thresh_cond = 1e6;
	double change = 1;
	double change2 = 0.1;
	cv::Vec2d err_std;

	cv::Matx33d _K;
	cv::Vec6d _D;
	if (flags & cv::fisheye::CALIB_USE_INTRINSIC_GUESS)
	{
		K.getMat().convertTo(_K, CV_64FC1);
		D.getMat().convertTo(_D, CV_64FC1);
		finalParam.Init(cv::Vec2d(_K(0, 0), _K(1, 1)),
			cv::Vec2d(_K(0, 2), _K(1, 2)),
			cv::Vec6d(flags & cv::fisheye::CALIB_FIX_K1 ? 0 : _D[0],
				flags & cv::fisheye::CALIB_FIX_K2 ? 0 : _D[1],
				flags & cv::fisheye::CALIB_FIX_K3 ? 0 : _D[2],
				flags & cv::fisheye::CALIB_FIX_K4 ? 0 : _D[3],
				_D[4], _D[5]),
			_K(0, 1) / _K(0, 0));
	}
	else
	{
		finalParam.Init(cv::Vec2d(std::max(image_size.width, image_size.height) / CV_PI, std::max(image_size.width, image_size.height) / CV_PI),
			cv::Vec2d(image_size.width / 2.0 - 0.5, image_size.height / 2.0 - 0.5));
	}

	errors.isEstimate = finalParam.isEstimate;

	std::vector<cv::Vec3d> omc(objectPoints.total()), Tc(objectPoints.total());

	CalibrateExtrinsics(objectPoints, imagePoints, finalParam, check_cond, thresh_cond, omc, Tc, RADIUS_D_FISHEYE_CALIB);


	//-------------------------------Optimization
	for (int iter = 0; iter < std::numeric_limits<int>::max(); ++iter)
	{
		if ((criteria.type == 1 && iter >= criteria.maxCount) ||
			(criteria.type == 2 && change <= criteria.epsilon && change2 <= criteria.epsilon) ||
			(criteria.type == 3 && ((change <= criteria.epsilon && change2 <= criteria.epsilon) || iter >= criteria.maxCount)))
			break;

		double alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, iter + 1.0);

		cv::Mat JJ2, ex3;
		ComputeJacobians(objectPoints, imagePoints, finalParam, omc, Tc, check_cond, thresh_cond, JJ2, ex3, RADIUS_D_FISHEYE_CALIB);

		int count = 1;
		cv::Mat G;
		int a = solve(JJ2, ex3, G, cv::DECOMP_LU);
		while(a == 0)
		{
			switch (count)
			{
			case 1:
				a = cv::solve(JJ2, ex3, G, cv::DECOMP_CHOLESKY);
				count++;
				break;
			case 2:
				cv::solve(JJ2, ex3, G, cv::DECOMP_SVD);
				a = 1;
				count++;
				break;
			case 3:
				a = cv::solve(JJ2, ex3, G, cv::DECOMP_QR);
				count++;
				break;
			case 4:
				a = cv::solve(JJ2, ex3, G, cv::DECOMP_EIG);
				count++;
				break;
			case 5:
				a = cv::solve(JJ2, ex3, G, cv::DECOMP_NORMAL);
				a = 1;
				count++;
				break;
			}
		}
		currentParam = finalParam + alpha_smooth2 * G;

		//for(int image_idx = 0; image_idx < objectPoints.total(); image_idx++)
		//{
		//	cv::Mat image, object;
		//	objectPoints.getMat(image_idx).convertTo(object, CV_64FC3);
		//	imagePoints.getMat(image_idx).convertTo(image, CV_64FC2);

		//	bool imT = image.rows < image.cols;
		//	cv::Mat om(omc[image_idx]), T(Tc[image_idx]);

		//	std::vector<cv::Point2d> x;
		//	cv::Mat jacobians;
		//	my_cv::internal::projectPoints(object, x, om, T, finalParam, jacobians, RADIUS_D_FISHEYE_CALIB);
		//	cv::Mat exkk = (imT ? image.t() : image) - cv::Mat(x);

		//	std::vector<cv::Point2d> x_cur;
		//	cv::Mat jacobians_cur;
		//	my_cv::internal::projectPoints(object, x_cur, om, T, currentParam, jacobians_cur, RADIUS_D_FISHEYE_CALIB);
		//	cv::Mat exkk_cur = (imT ? image.t() : image) - cv::Mat(x_cur);
		//}


		change = norm(cv::Vec4d(currentParam.f[0], currentParam.f[1], currentParam.c[0], currentParam.c[1]) -
			cv::Vec4d(finalParam.f[0], finalParam.f[1], finalParam.c[0], finalParam.c[1]))
			/ norm(cv::Vec4d(currentParam.f[0], currentParam.f[1], currentParam.c[0], currentParam.c[1]));
		change2 = norm(currentParam.k - finalParam.k) / norm(currentParam.k);

		if (a != 0)
		{
			finalParam = currentParam;
		}

		if (recompute_extrinsic)
		{
			CalibrateExtrinsics(objectPoints, imagePoints, finalParam, check_cond,
				thresh_cond, omc, Tc, RADIUS_D_FISHEYE_CALIB);
		}
	}

	//-------------------------------Validation
	double rms;
	EstimateUncertainties(objectPoints, imagePoints, finalParam, omc, Tc, errors, err_std, thresh_cond,
		check_cond, rms, RADIUS_D_FISHEYE_CALIB);

	//-------------------------------
	_K = cv::Matx33d(finalParam.f[0], finalParam.f[0] * finalParam.alpha, finalParam.c[0],
		0, finalParam.f[1], finalParam.c[1],
		0, 0, 1);

	if (K.needed()) cv::Mat(_K).convertTo(K, K.empty() ? CV_64FC1 : K.type());
	if (D.needed()) cv::Mat(finalParam.k).convertTo(D, D.empty() ? CV_64FC1 : D.type());
	if (rvecs.isMatVector())
	{
		int N = (int)objectPoints.total();

		if (rvecs.empty())
			rvecs.create(N, 1, CV_64FC3);

		if (tvecs.empty())
			tvecs.create(N, 1, CV_64FC3);

		for (int i = 0; i < N; i++)
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

double my_cv::fisheye_r_d::stereoCalibrate(cv::InputArrayOfArrays objectPoints, cv::InputArrayOfArrays imagePoints1, cv::InputArrayOfArrays imagePoints2,
	cv::InputOutputArray K1, cv::InputOutputArray D1, cv::InputOutputArray K2, cv::InputOutputArray D2, cv::Size imageSize,
	cv::OutputArray R, cv::OutputArray T, int flags, cv::TermCriteria criteria)
{

	CV_Assert(!objectPoints.empty() && !imagePoints1.empty() && !imagePoints2.empty());
	CV_Assert(objectPoints.total() == imagePoints1.total() || imagePoints1.total() == imagePoints2.total());
	CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
	CV_Assert(imagePoints1.type() == CV_32FC2 || imagePoints1.type() == CV_64FC2);
	CV_Assert(imagePoints2.type() == CV_32FC2 || imagePoints2.type() == CV_64FC2);

	CV_Assert(K1.empty() || (K1.size() == cv::Size(3, 3)));
	CV_Assert(D1.empty() || (D1.total() == 6));
	CV_Assert(K2.empty() || (K1.size() == cv::Size(3, 3)));
	CV_Assert(D2.empty() || (D1.total() == 6));

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
	cv::Vec6d _D1, _D2;
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

	intrinsicLeft.Init(cv::Vec2d(_K1(0, 0), _K1(1, 1)), cv::Vec2d(_K1(0, 2), _K1(1, 2)),
		cv::Vec6d(_D1[0], _D1[1], _D1[2], _D1[3], _D1[4], _D1[5]), _K1(0, 1) / _K1(0, 0));

	intrinsicRight.Init(cv::Vec2d(_K2(0, 0), _K2(1, 1)), cv::Vec2d(_K2(0, 2), _K2(1, 2)),
		cv::Vec6d(_D2[0], _D2[1], _D2[2], _D2[3], _D2[4], _D2[5]), _K2(0, 1) / _K2(0, 0));

	if ((flags & cv::fisheye::CALIB_FIX_INTRINSIC))
	{
		my_cv::internal::CalibrateExtrinsics(objectPoints, imagePoints1, intrinsicLeft, check_cond, thresh_cond, rvecs1, tvecs1, RADIUS_D_FISHEYE_CALIB);
		my_cv::internal::CalibrateExtrinsics(objectPoints, imagePoints2, intrinsicRight, check_cond, thresh_cond, rvecs2, tvecs2, RADIUS_D_FISHEYE_CALIB);
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
	intrinsicLeft.isEstimate[9] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
	intrinsicLeft.isEstimate[10] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;

	intrinsicRight.isEstimate[0] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
	intrinsicRight.isEstimate[1] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
	intrinsicRight.isEstimate[2] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
	intrinsicRight.isEstimate[3] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
	intrinsicRight.isEstimate[4] = flags & (cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
	intrinsicRight.isEstimate[5] = flags & (cv::fisheye::CALIB_FIX_K1 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
	intrinsicRight.isEstimate[6] = flags & (cv::fisheye::CALIB_FIX_K2 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
	intrinsicRight.isEstimate[7] = flags & (cv::fisheye::CALIB_FIX_K3 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
	intrinsicRight.isEstimate[8] = flags & (cv::fisheye::CALIB_FIX_K4 | cv::fisheye::CALIB_FIX_INTRINSIC) ? 0 : 1;
	intrinsicRight.isEstimate[9] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;
	intrinsicRight.isEstimate[10] = flags & cv::fisheye::CALIB_FIX_INTRINSIC ? 0 : 1;

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
	cv::Vec3d Tcur = my_cv::internal::median3d(T_list);

	cv::Mat J = cv::Mat::zeros(4 * n_points * n_images, 22 + 6 * (n_images + 1), CV_64FC1),
		e = cv::Mat::zeros(4 * n_points * n_images, 1, CV_64FC1), Jkk, ekk;

	for (int iter = 0; ; ++iter)
	{
		if ((criteria.type == 1 && iter >= criteria.maxCount) ||
			(criteria.type == 2 && change <= criteria.epsilon) ||
			(criteria.type == 3 && (change <= criteria.epsilon || iter >= criteria.maxCount)))
			break;

		J.create(4 * n_points * n_images, 22 + 6 * (n_images + 1), CV_64FC1);
		e.create(4 * n_points * n_images, 1, CV_64FC1);
		Jkk.create(4 * n_points, 22 + 6 * (n_images + 1), CV_64FC1);
		ekk.create(4 * n_points, 1, CV_64FC1);

		cv::Mat omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT;

		for (int image_idx = 0; image_idx < n_images; ++image_idx)
		{
			Jkk = cv::Mat::zeros(4 * n_points, 22 + 6 * (n_images + 1), CV_64FC1);

			cv::Mat object = objectPoints.getMat(image_idx).clone();
			cv::Mat imageLeft = imagePoints1.getMat(image_idx).clone();
			cv::Mat imageRight = imagePoints2.getMat(image_idx).clone();
			cv::Mat jacobians, projected;

			//left camera jacobian
			cv::Mat rvec = cv::Mat(rvecs1[image_idx]);
			cv::Mat tvec = cv::Mat(tvecs1[image_idx]);
			my_cv::internal::projectPoints(object, projected, rvec, tvec, intrinsicLeft, jacobians, RADIUS_D_FISHEYE_CALIB);
			cv::Mat(cv::Mat((imageLeft - projected).t()).reshape(1, 1).t()).copyTo(ekk.rowRange(0, 2 * n_points));
			jacobians.colRange(10, 13).copyTo(Jkk.colRange(28 + image_idx * 6, 31 + image_idx * 6).rowRange(0, 2 * n_points));
			jacobians.colRange(13, 16).copyTo(Jkk.colRange(31 + image_idx * 6, 34 + image_idx * 6).rowRange(0, 2 * n_points));
			jacobians.colRange(0, 2).copyTo(Jkk.colRange(0, 2).rowRange(0, 2 * n_points));
			jacobians.colRange(2, 4).copyTo(Jkk.colRange(2, 4).rowRange(0, 2 * n_points));
			jacobians.colRange(4, 10).copyTo(Jkk.colRange(5, 11).rowRange(0, 2 * n_points));
			jacobians.col(16).copyTo(Jkk.col(4).rowRange(0, 2 * n_points));

			//right camera jacobian
			my_cv::internal::compose_motion(rvec, tvec, omcur, Tcur, omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT);
			rvec = cv::Mat(rvecs2[image_idx]);
			tvec = cv::Mat(tvecs2[image_idx]);

			my_cv::internal::projectPoints(object, projected, omr, Tr, intrinsicRight, jacobians, RADIUS_D_FISHEYE_CALIB);
			cv::Mat(cv::Mat((imageRight - projected).t()).reshape(1, 1).t()).copyTo(ekk.rowRange(2 * n_points, 4 * n_points));
			cv::Mat dxrdom = jacobians.colRange(10, 13) * domrdom + jacobians.colRange(13, 16) * dTrdom;
			cv::Mat dxrdT = jacobians.colRange(10, 13) * domrdT + jacobians.colRange(13, 16)* dTrdT;
			cv::Mat dxrdomckk = jacobians.colRange(10, 13) * domrdomckk + jacobians.colRange(13, 16) * dTrdomckk;
			cv::Mat dxrdTckk = jacobians.colRange(10, 13) * domrdTckk + jacobians.colRange(13, 16) * dTrdTckk;

			dxrdom.copyTo(Jkk.colRange(22, 25).rowRange(2 * n_points, 4 * n_points));
			dxrdT.copyTo(Jkk.colRange(25, 28).rowRange(2 * n_points, 4 * n_points));
			dxrdomckk.copyTo(Jkk.colRange(28 + image_idx * 6, 31 + image_idx * 6).rowRange(2 * n_points, 4 * n_points));
			dxrdTckk.copyTo(Jkk.colRange(31 + image_idx * 6, 34 + image_idx * 6).rowRange(2 * n_points, 4 * n_points));
			jacobians.colRange(0, 2).copyTo(Jkk.colRange(11 + 0, 11 + 2).rowRange(2 * n_points, 4 * n_points));
			jacobians.colRange(2, 4).copyTo(Jkk.colRange(11 + 2, 11 + 4).rowRange(2 * n_points, 4 * n_points));
			jacobians.colRange(4, 10).copyTo(Jkk.colRange(11 + 5, 11 + 11).rowRange(2 * n_points, 4 * n_points));
			jacobians.col(16).copyTo(Jkk.col(11 + 4).rowRange(2 * n_points, 4 * n_points));

			//check goodness of sterepair
			double abs_max = 0;
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
		0, intrinsicLeft.f[1], intrinsicLeft.c[1],
		0, 0, 1);

	_K2 = cv::Matx33d(intrinsicRight.f[0], intrinsicRight.f[0] * intrinsicRight.alpha, intrinsicRight.c[0],
		0, intrinsicRight.f[1], intrinsicRight.c[1],
		0, 0, 1);

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

