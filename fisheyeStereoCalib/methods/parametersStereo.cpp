#include "parametersStereo.h"

using namespace cv;

bool check_image(const cv::Mat &image, std::string name)
{
	if (!image.data)
	{
		std::cerr << name << " data not loaded.\n";
		return false;
	}
	return true;
}


bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2)
{
	if (img1.cols != img2.cols || img1.rows != img2.rows)
	{
		std::cerr << "Images' dimensions do not corresponds.";
		return false;
	}
	return true;
}

/**
* @brief convert cv::Point2d(in OpenCV) to cv::Point(in OpenCV)
* @param point
* @return
*/
Point2d PointF2D(Point2f point)
{
	Point2d pointD = Point2d(point.x, point.y);
	return  pointD;
}
/**
 * \brief convert std::vector<cv::Point> to std::vector<cv::Point2d>
 * \param pts
 * \return
 */
std::vector<Point2d> VecPointF2D(std::vector<Point2f> pts)
{
	std::vector<Point2d> ptsD;
	for (std::vector<Point2f>::iterator iter = pts.begin(); iter != pts.end(); ++iter) {
		ptsD.push_back(PointF2D(*iter));
	}
	return ptsD;
}

/**
 * \brief to see how line distorted under certain camera model//from the spherical surface to image surface
 * \param mode 
 */
void fisheyeModel_show(camMode mode)
{
	cv::Mat img = cv::Mat(2000, 2560, CV_8UC1, Scalar::all(255));
	double R = 720;
	double R2 = R * R;
	double y_step = 1;
	for(int theta = -89; theta <= 89; )
	{
		double theta_ = tan(theta * PI / 180.0);
		double theta_2 = theta_ * theta_;
		for(double y_c = -R;  y_c <= R; )
		{
			double z_c = (R2 - y_c * y_c) / (1 + theta_2);
			z_c = sqrt(z_c);
			double x_c = theta_ * z_c;

			cv::Vec3d X_c = cv::Vec3d(x_c, y_c, z_c);

			if (X_c[2] < DBL_MIN)
				X_c[2] = 1.0;
			cv::Vec2d new_Xc = cv::Vec2d(X_c[0] / X_c[2], X_c[1] / X_c[2]);
			double new_r_2 = new_Xc.dot(new_Xc);
			double new_r_ = sqrt(new_r_2);
			double new_theta = atan(new_r_);
			double new_r;
			switch (mode)
			{
			case STEREOGRAPHIC:
				new_r = 2 * tan(new_theta / 2.0);
				break;
			case EQUIDISTANCE:
				new_r = new_theta;
				break;
			case EQUISOLID:
				new_r = 2 * sin(new_theta / 2.0);
				break;
			case ORTHOGONAL:
				new_r = sin(new_theta);
				break;
			case IDEAL_PERSPECTIVE:
				new_r = tan(new_theta);
				break;
			}
			cv::Vec2d new_pu = new_r / new_r_ * new_Xc;
			cv::Vec2d xd3(new_pu[0] , new_pu[1]);
			cv::Vec2d fi(xd3[0] * 360 + 1280, xd3[1] * 360 + 1000);

			int x = fi[0];
			int y = fi[1];
			if(x >= 0 && x < 2560 && y >= 0 && y < 2000)
			{
				img.at<uchar>(y, x) = 0;
			}
			

			y_c += y_step;
		}
		theta += 8.9;
	}

	cv::imwrite("img.jpg", img);
}

/**
 * \brief from world space to image //the world space coordinate is the same with the camera space coordinate
 * \param mode 
 */
void fisheyeModel_show2(camMode mode)
{
	int width = 1600;
	int height = 1440;
	cv::Mat img = cv::Mat(height, width, CV_8UC1, Scalar::all(255));
	int c_x = width / 2;
	int c_y = height / 2;
	double R = 720;
	double x_step = 100;
	double y_step = 100;
	int z = R;
	int bound = 2 * R;
	for(int x = -bound; x <= bound; )
	{
		for(int y = -50 * bound; y <= 50 * bound; y++)
		{
			cv::Vec3d X_c = cv::Vec3d(x, y, z);

			if (X_c[2] < DBL_MIN)
				X_c[2] = 1.0;
			cv::Vec2d new_Xc = cv::Vec2d(X_c[0] / X_c[2], X_c[1] / X_c[2]);
			double new_r_2 = new_Xc.dot(new_Xc);
			double new_r_ = sqrt(new_r_2);
			double new_theta = atan(new_r_);
			double new_r;
			switch (mode)
			{
			case STEREOGRAPHIC:
				new_r = 2 * tan(new_theta / 2.0);
				break;
			case EQUIDISTANCE:
				new_r = new_theta;
				break;
			case EQUISOLID:
				new_r = 2 * sin(new_theta / 2.0);
				break;
			case ORTHOGONAL:
				new_r = sin(new_theta);
				break;
			case IDEAL_PERSPECTIVE:
				new_r = tan(new_theta);
				break;
			}
			cv::Vec2d new_pu = new_r / new_r_ * new_Xc;
			cv::Vec2d xd3(new_pu[0], new_pu[1]);
			cv::Vec2d fi(xd3[0] * 400 + c_x, xd3[1] * 360 + c_y);

			int u = fi[0];
			int v = fi[1];
			if (u >= 0 && u < width && v >= 0 && v < height)
			{
				img.at<uchar>(v, u) = 0;
			}

		}
		x += x_step;
	}

	for (int y = -bound; y <= bound; )
	{
		for (int x = -50 * bound; x <= 50 * bound; x++)
		{
			cv::Vec3d X_c = cv::Vec3d(x, y, z);

			if (X_c[2] < DBL_MIN)
				X_c[2] = 1.0;
			cv::Vec2d new_Xc = cv::Vec2d(X_c[0] / X_c[2], X_c[1] / X_c[2]);
			double new_r_2 = new_Xc.dot(new_Xc);
			double new_r_ = sqrt(new_r_2);
			double new_theta = atan(new_r_);
			double new_r;
			switch (mode)
			{
			case STEREOGRAPHIC:
				new_r = 2 * tan(new_theta / 2.0);
				break;
			case EQUIDISTANCE:
				new_r = new_theta;
				break;
			case EQUISOLID:
				new_r = 2 * sin(new_theta / 2.0);
				break;
			case ORTHOGONAL:
				new_r = sin(new_theta);
				break;
			case IDEAL_PERSPECTIVE:
				new_r = tan(new_theta);
				break;
			}
			cv::Vec2d new_pu = new_r / new_r_ * new_Xc;
			cv::Vec2d xd3(new_pu[0], new_pu[1]);
			cv::Vec2d fi(xd3[0] * 400 + c_x, xd3[1] * 360 + c_y);

			int u = fi[0];
			int v = fi[1];
			if (u >= 0 && u < width && v >= 0 && v < height)
			{
				img.at<uchar>(v, u) = 0;
			}

		}
		y += y_step;
	}

	cv::imwrite("img.jpg", img);
}

void fisheyeModel_show3(camMode mode)
{

}
