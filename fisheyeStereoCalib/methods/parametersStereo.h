#pragma once
#include <opencv2/opencv.hpp>


#define IMG_CAPTURE_WIDTH	2560
#define IMG_CAPTURE_HEIGHT	1440

typedef  std::vector<std::vector<cv::Point2f> >  douVecPt2f;
typedef  std::vector<std::vector<cv::Point3f> >  douVecPt3f;

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

enum PCLFILTERS_
{
	PASS_THROUGH = 0,					//ʹ��ֱͨ�˲����Ե��ƽ����˲�����
	VOXEL_GRID = 1,						//ʹ��VoxelGrid�˲��������²����������������е��������������ʾ������������
	STATISTIC_OUTLIERS_REMOVE = 2,		//ʹ��StatisticalOutlierRemoval�˲����Ƴ���Ⱥ��
	MODEL_COEFFICIENTS = 3,				//����ͶӰ��һ��������ģ���ϣ�����ƽ�����ȣ�
	EXTRACT_INDICES = 4,				//ʹ��ExtractIndices�˲���������ĳһ�ָ��㷨��ȡ�����е�һ���Ӽ�
	CONDITIONAL_REMOVAL = 5,			//ʹ��CondidtionalRemoval�˲�����һ��ɾ�����������ĵ����趨��һ����������ָ����������ݵ�
	RADIUS_OUTLIER_REMOVAL = 6,			//ʹ��RadiusOutlierRemoval�˲�����ɾ��������ĵ���һ����Χ��û�дﵽ�㹻����������������ݵ�
	CROP_HULL= 7,						//ʹ��CropHull�˲����õ�2D��ն�����ڲ������ⲿ�ĵ���
};

enum CONSENSUS_MODEL_TYPE_
{
	CONSENSUS_MODEL_SPHERE_ = 0,
	CONSENSUS_MODEL_PLANE_ = 1,
};

//fisheye image correction///////from globalInclude.h
//typedef cv::Mat			Mat;
//typedef cv::Point		Point;
//typedef cv::Point2i		Point2i;
//typedef cv::Size		Size;
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
	ORTHOGONAL
}camMode;

typedef enum
{
	PERSPECTIVE,
	LATITUDE_LONGTITUDE,
}distMapMode;



//
bool check_image(const cv::Mat &image, std::string name = "Image");
bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2);

cv::Point2d PointF2D(cv::Point2f point);
std::vector<cv::Point2d> VecPointF2D(std::vector<cv::Point2f> pts);


template<typename _Tp>
std::vector<_Tp> convertMat2Vector(const cv::Mat mat)
{
	if (mat.isContinuous())
	{
		return (std::vector<_Tp>)(mat.reshape(0, 1));
	}

	cv::Mat mat_ = mat.clone();
	std::vector<_Tp> vecMat = mat_.reshape(0, 1);
	return (std::vector<_Tp>)(mat_.reshape(0, 1));
}