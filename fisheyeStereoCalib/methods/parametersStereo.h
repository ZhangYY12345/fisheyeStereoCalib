#pragma once
#include <opencv2/opencv.hpp>


#define IMG_CAPTURE_WIDTH	2560
#define IMG_CAPTURE_HEIGHT	1440

typedef  std::vector<std::vector<cv::Point2f> >  douVecPt2f;
typedef  std::vector<std::vector<cv::Point3f> >  douVecPt3f;
typedef  std::vector<std::vector<cv::Point2d> >  douVecPt2d;
typedef  std::vector<std::vector<cv::Point3d> >  douVecPt3d;


enum DisparityType
{
	DISPARITY_LEFT = 0,
	DISPARITY_RIGHT = 1,
};

enum StereoMatchingAlgorithms
{
	BM = 0,
	SGBM = 1,
	ADAPTIVE_WEIGHT = 2,
	ADAPTIVE_WEIGHT_8DIRECT = 3,
	ADAPTIVE_WEIGHT_GEODESIC = 4,
	ADAPTIVE_WEIGHT_BILATERAL_GRID = 5,
	ADAPTIVE_WEIGHT_BLO1 = 6,
	ADAPTIVE_WEIGHT_GUIDED_FILTER = 7,
	ADAPTIVE_WEIGHT_GUIDED_FILTER_2 = 8,
	ADAPTIVE_WEIGHT_GUIDED_FILTER_3 = 9,
	ADAPTIVE_WEIGHT_MEDIAN = 10,
};

enum PCL_FILTERS_
{
	PASS_THROUGH = 0,					//使用直通滤波器对点云进行滤波处理
	VOXEL_GRID = 1,						//使用VoxelGrid滤波器进行下采样：用体素内所有点的重心来近似显示体素中其他点
	STATISTIC_OUTLIERS_REMOVE = 2,		//使用StatisticalOutlierRemoval滤波器移除离群点
	MODEL_COEFFICIENTS = 3,				//将点投影到一个参数化模型上（例，平面或球等）
	EXTRACT_INDICES = 4,				//使用ExtractIndices滤波器，基于某一分割算法提取点云中的一个子集
	CONDITIONAL_REMOVAL = 5,			//使用CondidtionalRemoval滤波器，一次删除满足对输入的点云设定的一个或多个条件指标的所有数据点
	RADIUS_OUTLIER_REMOVAL = 6,			//使用RadiusOutlierRemoval滤波器，删除在输入的点云一定范围内没有达到足够多近邻数的所有数据点
	CROP_HULL= 7,						//使用CropHull滤波器得到2D封闭多边形内部或者外部的点云
};

enum CONSENSUS_MODEL_TYPE_
{
	CONSENSUS_MODEL_SPHERE_ = 0,
	CONSENSUS_MODEL_PLANE_ = 1,
};

//fisheye image correction///////from globalInclude.h
//typedef cv::Mat			cv::Mat;
//typedef cv::Point		Point;
//typedef cv::Point2i		Point2i;
//typedef cv::Size		cv::Size;
//typedef cv::Vec3b		Vec3b;
//typedef cv::Scalar		Scalar;
//typedef cv::Rect		Rect;
//typedef cv::Point3f		Point3f;
//typedef cv::Point3i		Point3i;
//typedef cv::Stitcher	Stitcher;

const double  PI = 3.1415926535897932384626433832795;
const double  LIMIT = 1e-4;

enum CorrectType
{
	Forward,
	//means correct the distorted image by mapping the pixels on the origin image
	//to the longitude-latitude rectified image, there may be some pixels on the
	//rectified image which have no corresponding origin pixel. 
	Reverse,
	//means correct the distorted image by reverse mapping, that is from the rectified 
	//image to the origin distorted image, this method can be sure for that every pixels
	//on the rectified image have its corresponding origin pixel.
};

typedef enum
{
	STEREOGRAPHIC,
	EQUIDISTANCE,
	EQUISOLID,
	ORTHOGONAL,
	IDEAL_PERSPECTIVE,
}camMode;

typedef enum
{
	PERSPECTIVE,
	LATITUDE_LONGTITUDE,	//经纬度
}distMapMode;

typedef  enum
{
	ORIGIN_FISHEYE_CALIB,
	THETA_D_FISHEYE_CALIB,
	RADIUS_D_FISHEYE_CALIB,
	RADIUS_RD_FISHEYE_CALIB,
	RADIUS_RD2_FISHEYE_CALIB,
}DISTORT_Mode_Fisheye;

typedef enum
{
	TOP_LEFT,
	TOP_RIGHT,
	BOTTOM_LEFT,
	BOTTOM_RIGHT,
}RIGHT_COUNT_SIDE;
//
bool check_image(const cv::Mat &image, std::string name = "Image");
bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2);

cv::Point2d PointF2D(cv::Point2f point);
std::vector<cv::Point2d> VecPointF2D(std::vector<cv::Point2f> pts);


void fisheyeModel_show(camMode mode);
void fisheyeModel_show2(camMode mode);
void fisheyeModel_show3(camMode mode);

template<typename _Tp>
std::vector<_Tp> convertMat2Vector(const cv::Mat mat)
{
	if (mat.isContinuous())
	{
		return (std::vector<_Tp>)(mat.reshape(0, 1));
	}

	cv::Mat mat_ = mat.clone();
	std::vector<_Tp> veccv::Mat = mat_.reshape(0, 1);
	return (std::vector<_Tp>)(mat_.reshape(0, 1));
}