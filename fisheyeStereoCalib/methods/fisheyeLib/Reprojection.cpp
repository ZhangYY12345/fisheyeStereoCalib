//
//  Reprojection.cpp
//  Reprojection
//
//  Created by Ryohei Suda on 2014/09/12.
//  Copyright (c) 2014年 RyoheiSuda. All rights reserved.
//

#include "Reprojection.h"
#include <sstream>

void Reprojection::loadParameters(std::string filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    
    double f, f0;
    cv::Point2d center;
    cv::Size2i img_size;
    std::vector<double> a;
    
    if (!fs.isOpened()) {
        std::cerr << filename << " cannnot be opened!" << std::endl;
        exit(-1);
    }
    fs["f"] >> f;
    fs["f0"] >> f0;
    fs["center"] >> center;
    fs["img_size"] >> img_size;
    fs["projection"] >> projection;
    
    a.clear();
    cv::FileNode fn = fs["a"];
    cv::FileNodeIterator it = fn.begin();
    for (; it != fn.end(); ++it) {
        a.push_back((static_cast<double>(*it)));
    }
    
    IncidentVector::setProjection(projection);
    IncidentVector::setParameters(f, f0, a, img_size, center);
}


void Reprojection::theta2radius()
{
    double max_r = 0;
    // Calculate the longest distance between optical center and image corner
    if (IncidentVector::getImgSize().width - IncidentVector::getCenter().x > IncidentVector::getImgSize().width / 2.0) {
        max_r += pow(IncidentVector::getImgSize().width - IncidentVector::getCenter().x, 2);
    } else {
        max_r += pow(IncidentVector::getCenter().x, 2);
    }
    if (IncidentVector::getImgSize().height - IncidentVector::getCenter().y > IncidentVector::getImgSize().height / 2.0) {
        max_r += pow(IncidentVector::getImgSize().height - IncidentVector::getCenter().y, 2);
    } else {
        max_r += pow(IncidentVector::getCenter().y, 2);
    }
    max_r = sqrt(max_r);
    //max_r = 2000;
    int theta_size = (round(max_r) + 1) * precision; // If PRECISION = 10, r = {0, 0.1, 0.2, 0.3, ...}，查找表中距离r的离散取值的个数，每个距离r对应一个入射角
	ideal_r = round(max_r) + 1;

    r2t.resize(theta_size);
    cv::Point2d p(0,0);
    IncidentVector *iv = nullptr;
    std::cout << IncidentVector::getProjectionName()  << "\t" << IncidentVector::getProjection()<< std::endl;
    switch (IncidentVector::getProjection()) {
        case 0:
            iv = new StereographicProjection(p);
            std::cout << "Stereographic Projection" << std::endl;
            break;
        case 1:
            iv = new OrthographicProjection(p);
            std::cout << "Orthographic Projection" << std::endl;
            break;
        case 2:
            iv = new EquidistanceProjection(p);
            std::cout << "Equidistance Projection" << std::endl;
            break;
        case 3:
            iv = new EquisolidAngleProjection(p);
            std::cout << "Equisolid Angle Projection" << std::endl;
            break;
    }

	if(IncidentVector::getProjection() == 0 || IncidentVector::getProjection() == 2)
	{
		for (int i = 0; i < theta_size; ++i) {
			double r = (double)i / precision;

			r2t[i] = iv->aoi(r);//计算theta角，此处用到畸变公式
		}
	}
	else if (IncidentVector::getProjection() == 1 || IncidentVector::getProjection() == 3) // notice :if t in asin(t) larger than 1, the theta is nan;
	{
		for (int i = 0; i < theta_size; ++i) {
			double r = (double)i / precision;

			r2t[i] = iv->aoi(r);//计算theta角，此处用到畸变公式
			if (r2t[i] == CV_2PI)
			{
				r2t[i] = r2t[i - 1];
			}
		}

	}


    
	int r_size = (round(max_r) + 1) * precision; //查找表中距离r的离散取值的个数，每个距离r对应一个唯一的入射角
//    r_size = 2000 * precision;
    t2r.resize(r_size);
    int j = 1; // j/PRECISION: radius
    rad_step = r2t[theta_size-1] / (double)r_size; // 0 ~ theta[end] radian  ：r2t[theta_size-1]对应最大的入射角；rad_step:入射角离散取值的间隔
    //rad_step = 2.35 / r_size;
    for (int i = 0; i < r_size; ++i) {
        double rad = rad_step * i; //入射角取值为rad时，对应点在一圆上
        for (; j < theta_size; ++j) { //对不同的距离r
            if (r2t[j] > rad) { //当距离j对应的入射角r2t[j]第一次大于rad时
                t2r[i] = ((i*rad_step - r2t[j-1]) / (r2t[j]-r2t[j-1]) + j-1) / precision; // See my note on 2014/6/6
                break;
            }
        }
    }
}

void Reprojection::saveTheta2Radius(std::string filename)
{
    std::ofstream ofs(filename);
    
    for (int i = 0; i < t2r.size(); ++i) {
        ofs << t2r[i] << ' ' << rad_step * i << ' ' << rad_step * i * 180 / M_PI << std::endl;
    }
    
    ofs.close();
}

void Reprojection::loadTheta2Radius(std::string filename)
{
	std::ifstream inFile(filename);

	if(!inFile)
	{
		std::cout << "fail to open .dat file...";
		return;
	}

	t2r.clear();

	std::stringstream ss;
	std::string temp;
	while (std::getline(inFile, temp))
	{
		ss.clear();
		ss.str(temp);

		double val;
		ss >> val;
		t2r.push_back(val);
	}
}

void Reprojection::saveRadius2Theta(std::string filename)
{
    std::ofstream ofs(filename);

    for (int i = 0; i < r2t.size(); ++i) {
        ofs << (double)i/precision << ' ' << r2t[i] << ' ' << r2t[i] * 180 / M_PI << std::endl;
    }
    
    ofs.close();
}

void Reprojection::loadRadius2Theta(std::string filename)
{
	std::ifstream inFile(filename);

	if (!inFile)
	{
		std::cout << "fail to open .dat file...";
		return;
	}

	r2t.clear();

	std::stringstream ss;
	std::string temp;
	while (std::getline(inFile, temp))
	{
		ss.clear();
		ss.str(temp);

		double val;
		ss >> val;
		ss >> val;
		r2t.push_back(val);
	}
}

/**
 * \brief save t2r[], ideal_r and center for the undistorted image into XML file
 * \param fileName 
 */
void Reprojection::saveReprojectData(std::string fileName, bool isSave_t2r)
{
	cv::FileStorage fn(fileName, cv::FileStorage::WRITE);
	fn << "Radius" << ideal_r;
	fn << "Center_x" << ideal_center.x;
	fn << "Center_y" << ideal_center.y;

	if(isSave_t2r)
	{
		std::string t2rFileName = fileName.substr(0, fileName.length() - 4) + "_t2r.dat";
		saveTheta2Radius(t2rFileName);
		fn << "t2r_fileName" << t2rFileName;
	}

	fn.release();
}


/**
 * \brief 基于透视投影模型的map计算，（理想情况下的入射角计算为透视投影模型）
 * \param theta_x 
 * \param theta_y 
 * \param f_ 
 * \param mapx 
 * \param mapy 
 */
void Reprojection::calcMaps(double theta_x, double theta_y, double f_, cv::Mat& mapx, cv::Mat& mapy)
{
    mapx.create(IncidentVector::getImgSize().height, IncidentVector::getImgSize().width, CV_32FC1);
    mapy.create(IncidentVector::getImgSize().height, IncidentVector::getImgSize().width, CV_32FC1);

	ideal_center = cv::Point2d(IncidentVector::getImgSize().width / 2.0, IncidentVector::getImgSize().height / 2.0);

    cv::Mat Rx = (cv::Mat_<double>(3,3) <<
                  1,            0,             0,
                  0, cos(theta_x), -sin(theta_x),
                  0, sin(theta_x),  cos(theta_x));
    cv::Mat Ry = (cv::Mat_<double>(3,3) <<
                  cos(theta_y),  0, sin(theta_y),
                  0,             1,            0,
                  -sin(theta_y), 0, cos(theta_y));
    cv::Mat R = Ry * Rx;
	
	for (int y_ = 0; y_ < IncidentVector::getImgSize().height; ++y_) { // y
        for (int x_ = 0; x_ < IncidentVector::getImgSize().width; ++x_) { // x
            
            cv::Point2d p2(x_ - IncidentVector::getImgSize().width / 2.0, y_ - IncidentVector::getImgSize().height / 2.0);
            cv::Mat p3 = (cv::Mat_<double>(3,1) << p2.x, p2.y, f_);
            cv::Mat real = 1.0/sqrt(pow(p2.x,2) + pow(p2.y,2) + pow(f_,2)) * R * p3;
            
            double x = real.at<double>(0,0);
            double y = real.at<double>(1,0);
            double z = real.at<double>(2,0);
            double theta = atan2(sqrt(1-pow(z,2)), z);
            if (t2r.size() <= (int)(theta/rad_step)) {
                mapx.at<float>(y_,x_) = 0;
                mapy.at<float>(y_,x_) = 0;
                continue;
            }
            cv::Point2d final = IncidentVector::getCenter() + t2r[(int)(theta/rad_step)]  * (cv::Point2d(x,y) / sqrt(1-pow(z,2)));
            //            cv::Point2d final = center + f * theta / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Perspective projection
            //            cv::Point2d final = center + 2*f*tan(theta/2) / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Stereo graphic projection
            
            mapx.at<float>(y_,x_) = final.x;
            mapy.at<float>(y_,x_) = final.y;
        }
    }
}

void Reprojection::calcMaps(double f_, cv::Mat& mapx, cv::Mat& mapy)
{
	mapx.create(IncidentVector::getImgSize().height, IncidentVector::getImgSize().width, CV_32FC1);
	mapy.create(IncidentVector::getImgSize().height, IncidentVector::getImgSize().width, CV_32FC1);

	ideal_center = cv::Point2d(IncidentVector::getImgSize().width / 2.0, IncidentVector::getImgSize().height / 2.0);

	for (int y_ = 0; y_ < IncidentVector::getImgSize().height; ++y_) { // y
		for (int x_ = 0; x_ < IncidentVector::getImgSize().width; ++x_) { // x

			cv::Point2d p2(x_ - IncidentVector::getImgSize().width / 2.0, y_ - IncidentVector::getImgSize().height / 2.0);
			cv::Mat p3 = (cv::Mat_<double>(3, 1) << p2.x, p2.y, f_);
			cv::Mat real = 1.0 / sqrt(pow(p2.x, 2) + pow(p2.y, 2) + pow(f_, 2))  * p3;

			double x = real.at<double>(0, 0);
			double y = real.at<double>(1, 0);
			double z = real.at<double>(2, 0);
			double theta = atan2(sqrt(1 - pow(z, 2)), z);
			if (t2r.size() <= (int)(theta / rad_step)) {
				mapx.at<float>(y_, x_) = 0;
				mapy.at<float>(y_, x_) = 0;
				continue;
			}
			cv::Point2d final = IncidentVector::getCenter() + t2r[(int)(theta / rad_step)] / sqrt(1 - pow(z, 2)) * cv::Point2d(x, y);
			//            cv::Point2d final = center + f * theta / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Perspective projection
			//            cv::Point2d final = center + 2*f*tan(theta/2) / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Stereo graphic projection

			mapx.at<float>(y_, x_) = final.x;
			mapy.at<float>(y_, x_) = final.y;
		}
	}
}

/**
 * \brief using fisheye model to calculate the ideal incidence angle
 * \param theta_x 
 * \param theta_y 
 * \param f_ 
 * \param scale 
 * \param mapx 
 * \param mapy 
 */
void Reprojection::calcMaps_fisheye_model_full(double theta_x, double theta_y, double f_, cv::Mat& mapx,
	cv::Mat& mapy, int scale, bool isFisheyeModel)
{
	mapx.create(IncidentVector::getImgSize().height * scale, IncidentVector::getImgSize().width * scale, CV_32FC1);
	mapy.create(IncidentVector::getImgSize().height * scale, IncidentVector::getImgSize().width * scale, CV_32FC1);

	ideal_center = cv::Point2d(IncidentVector::getImgSize().width * scale / 2.0, IncidentVector::getImgSize().height * scale / 2.0);

	cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
		1, 0, 0,
		0, cos(theta_x), -sin(theta_x),
		0, sin(theta_x), cos(theta_x));
	cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
		cos(theta_y), 0, sin(theta_y),
		0, 1, 0,
		-sin(theta_y), 0, cos(theta_y));
	cv::Mat R = Ry * Rx;

	for (int y_ = 0; y_ < IncidentVector::getImgSize().height * scale; ++y_) { // y
		for (int x_ = 0; x_ < IncidentVector::getImgSize().width * scale; ++x_) { // x

			cv::Point2d p2(x_ - IncidentVector::getImgSize().width * scale / 2.0, y_ - IncidentVector::getImgSize().height * scale / 2.0);
			cv::Mat p3 = (cv::Mat_<double>(3, 1) << p2.x, p2.y, f_);
			cv::Mat real = 1.0 / sqrt(pow(p2.x, 2) + pow(p2.y, 2) + pow(f_, 2)) * R * p3;

			double x = real.at<double>(0, 0);
			double y = real.at<double>(1, 0);
			double z = real.at<double>(2, 0);
			double theta;
			if (isFisheyeModel)
			{
				switch (IncidentVector::getProjection())
				{
				case 0://StereographicProjection
					theta = 2 * atan2(sqrt(1 - pow(z, 2)), 2 * z);
					break;
				case 1://OrthographicProjection
					theta = asin(sqrt(1 - pow(z, 2)) / z);
					break;
				case 2://EquidistanceProjection
					theta = sqrt(1 - pow(z, 2)) / z;
					break;
				case 3://EquisolidAngleProjection
					theta = 2 * asin(sqrt(1 - pow(z, 2)) / (2 * z));
					break;
				default:
					theta = atan2(sqrt(1 - pow(z, 2)), z);
				}
			}
			else//透视投影
			{
				theta = atan2(sqrt(1 - pow(z, 2)), z);
			}

			if (t2r.size() <= (int)(theta / rad_step)) {
				mapx.at<float>(y_, x_) = 0;
				mapy.at<float>(y_, x_) = 0;
				continue;
			}
			cv::Point2d final = IncidentVector::getCenter() + t2r[(int)(theta / rad_step)] * (cv::Point2d(x, y) / sqrt(1 - pow(z, 2)));
			//            cv::Point2d final = center + f * theta / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Perspective projection
			//            cv::Point2d final = center + 2*f*tan(theta/2) / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Stereo graphic projection

			mapx.at<float>(y_, x_) = final.x;
			mapy.at<float>(y_, x_) = final.y;
		}
	}
}

void Reprojection::calcMaps_fisheye_model_full(double f_, cv::Mat& mapx, cv::Mat& mapy, int scale, bool isFisheyeModel)
{
	mapx.create(IncidentVector::getImgSize().height * scale, IncidentVector::getImgSize().width * scale, CV_32FC1);
	mapy.create(IncidentVector::getImgSize().height * scale, IncidentVector::getImgSize().width * scale, CV_32FC1);

	ideal_center = cv::Point2d(IncidentVector::getImgSize().width * scale / 2.0, IncidentVector::getImgSize().height * scale / 2.0);

	for (int y_ = 0; y_ < IncidentVector::getImgSize().height * scale; ++y_) { // y
		for (int x_ = 0; x_ < IncidentVector::getImgSize().width * scale; ++x_) { // x

			cv::Point2d p2(x_ - IncidentVector::getImgSize().width * scale / 2.0, y_ - IncidentVector::getImgSize().height * scale / 2.0);
			cv::Mat p3 = (cv::Mat_<double>(3, 1) << p2.x, p2.y, f_);
			cv::Mat real = 1.0 / sqrt(pow(p2.x, 2) + pow(p2.y, 2) + pow(f_, 2))  * p3;

			double x = real.at<double>(0, 0);
			double y = real.at<double>(1, 0);
			double z = real.at<double>(2, 0);
			double theta;
			if (isFisheyeModel)
			{
				switch (IncidentVector::getProjection())
				{
				case 0://StereographicProjection
					theta = 2 * atan2(sqrt(1 - pow(z, 2)), 2 * z);
					break;
				case 1://OrthographicProjection
					theta = asin(sqrt(1 - pow(z, 2)) / z);
					break;
				case 2://EquidistanceProjection
					theta = sqrt(1 - pow(z, 2)) / z;
					break;
				case 3://EquisolidAngleProjection
					theta = 2 * asin(sqrt(1 - pow(z, 2)) / 2.0 / z);
					break;
				default:
					theta = atan2(sqrt(1 - pow(z, 2)), z);
				}
			}
			else//透视投影
			{
				theta = atan2(sqrt(1 - pow(z, 2)), z);
			}

			if (t2r.size() <= (int)(theta / rad_step)) {
				mapx.at<float>(y_, x_) = 0;
				mapy.at<float>(y_, x_) = 0;
				continue;
			}
			cv::Point2d final = IncidentVector::getCenter() + t2r[(int)(theta / rad_step)] / sqrt(1 - pow(z, 2)) * cv::Point2d(x, y);
			//            cv::Point2d final = center + f * theta / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Perspective projection
			//            cv::Point2d final = center + 2*f*tan(theta/2) / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Stereo graphic projection

			mapx.at<float>(y_, x_) = final.x;
			mapy.at<float>(y_, x_) = final.y;
		}
	}
}

void Reprojection::calcMaps_fisheye_model_offset_full(double theta_x, double theta_y, double f_, cv::Mat& mapx,
	cv::Mat& mapy, int offset, bool isFisheyeModel)
{
	mapx.create(IncidentVector::getImgSize().height + offset * 2, IncidentVector::getImgSize().width + offset * 2, CV_32FC1);
	mapy.create(IncidentVector::getImgSize().height + offset * 2, IncidentVector::getImgSize().width + offset * 2, CV_32FC1);

	ideal_center = cv::Point2d(IncidentVector::getImgSize().width / 2.0 + offset, IncidentVector::getImgSize().height / 2.0 + offset);

	cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
		1, 0, 0,
		0, cos(theta_x), -sin(theta_x),
		0, sin(theta_x), cos(theta_x));
	cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
		cos(theta_y), 0, sin(theta_y),
		0, 1, 0,
		-sin(theta_y), 0, cos(theta_y));
	cv::Mat R = Ry * Rx;

	for (int y_ = 0; y_ < IncidentVector::getImgSize().height + offset * 2; ++y_) { // y
		for (int x_ = 0; x_ < IncidentVector::getImgSize().width + offset * 2; ++x_) { // x

			cv::Point2d p2(x_ - (IncidentVector::getImgSize().width + offset * 2) / 2.0, y_ - (IncidentVector::getImgSize().height + offset * 2) / 2.0);
			cv::Mat p3 = (cv::Mat_<double>(3, 1) << p2.x, p2.y, f_);
			cv::Mat real = 1.0 / sqrt(pow(p2.x, 2) + pow(p2.y, 2) + pow(f_, 2)) * R * p3;

			double x = real.at<double>(0, 0);
			double y = real.at<double>(1, 0);
			double z = real.at<double>(2, 0);
			double theta;
			if (isFisheyeModel)
			{
				switch (IncidentVector::getProjection())
				{
				case 0://StereographicProjection
					theta = 2 * atan2(sqrt(1 - pow(z, 2)), 2 * z);
					break;
				case 1://OrthographicProjection
					theta = asin(sqrt(1 - pow(z, 2)) / z);
					break;
				case 2://EquidistanceProjection
					theta = sqrt(1 - pow(z, 2)) / z;
					break;
				case 3://EquisolidAngleProjection
					theta = 2 * asin(sqrt(1 - pow(z, 2)) / (2 * z));
					break;
				default:
					theta = atan2(sqrt(1 - pow(z, 2)), z);
				}
			}
			else//透视投影
			{
				theta = atan2(sqrt(1 - pow(z, 2)), z);
			}

			if (t2r.size() <= (int)(theta / rad_step)) {
				mapx.at<float>(y_, x_) = 0;
				mapy.at<float>(y_, x_) = 0;
				continue;
			}
			cv::Point2d final = IncidentVector::getCenter() + t2r[(int)(theta / rad_step)] * (cv::Point2d(x, y) / sqrt(1 - pow(z, 2)));
			//            cv::Point2d final = center + f * theta / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Perspective projection
			//            cv::Point2d final = center + 2*f*tan(theta/2) / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Stereo graphic projection

			mapx.at<float>(y_, x_) = final.x;
			mapy.at<float>(y_, x_) = final.y;
		}
	}
}

void Reprojection::calcMaps_fisheye_model_offset_full(double f_, cv::Mat& mapx, cv::Mat& mapy, int offset, bool isFisheyeModel)
{
	mapx.create(IncidentVector::getImgSize().height + offset * 2, IncidentVector::getImgSize().width + offset * 2, CV_32FC1);
	mapy.create(IncidentVector::getImgSize().height + offset * 2, IncidentVector::getImgSize().width + offset * 2, CV_32FC1);

	ideal_center = cv::Point2d(IncidentVector::getImgSize().width / 2.0 + offset, IncidentVector::getImgSize().height / 2.0 + offset);

	for (int y_ = 0; y_ < IncidentVector::getImgSize().height + offset * 2; ++y_) { // y
		for (int x_ = 0; x_ < IncidentVector::getImgSize().width + offset * 2; ++x_) { // x

			cv::Point2d p2(x_ - (IncidentVector::getImgSize().width + offset * 2) / 2.0, y_ - (IncidentVector::getImgSize().height + offset * 2) / 2.0);
			cv::Mat p3 = (cv::Mat_<double>(3, 1) << p2.x, p2.y, f_);
			cv::Mat real = 1.0 / sqrt(pow(p2.x, 2) + pow(p2.y, 2) + pow(f_, 2)) * p3;

			double x = real.at<double>(0, 0);
			double y = real.at<double>(1, 0);
			double z = real.at<double>(2, 0);
			double theta;
			if (isFisheyeModel)
			{
				switch (IncidentVector::getProjection())
				{
				case 0://StereographicProjection
					theta = 2 * atan2(sqrt(1 - pow(z, 2)), 2 * z);
					break;
				case 1://OrthographicProjection
					theta = asin(sqrt(1 - pow(z, 2)) / z);
					break;
				case 2://EquidistanceProjection
					theta = sqrt(1 - pow(z, 2)) / z;
					break;
				case 3://EquisolidAngleProjection
					theta = 2 * asin(sqrt(1 - pow(z, 2)) / (2 * z));
					break;
				default:
					theta = atan2(sqrt(1 - pow(z, 2)), z);
				}
			}
			else//透视投影
			{
				theta = atan2(sqrt(1 - pow(z, 2)), z);
			}

			if (t2r.size() <= (int)(theta / rad_step)) {
				mapx.at<float>(y_, x_) = 0;
				mapy.at<float>(y_, x_) = 0;
				continue;
			}
			cv::Point2d final = IncidentVector::getCenter() + t2r[(int)(theta / rad_step)] * (cv::Point2d(x, y) / sqrt(1 - pow(z, 2)));
			//            cv::Point2d final = center + f * theta / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Perspective projection
			//            cv::Point2d final = center + 2*f*tan(theta/2) / sqrt(1-pow(z,2))  * cv::Point2d(x,y); // Stereo graphic projection

			mapx.at<float>(y_, x_) = final.x;
			mapy.at<float>(y_, x_) = final.y;
		}
	}
}
