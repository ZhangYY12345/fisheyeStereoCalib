#include "fisheyeCalib3d.h"
#include "fisheye_opencv/fisheyeCalib_theta_d.h"
#include "fisheye_opencv/fisheyeCalib_radius_d.h"
#include "fisheye_opencv/fisheyeCalib_radius_rd.h"
#include "fisheye_opencv/fisheyeCalib_raduis_rd2.h"
#include <stack>
#include "fisheyeLib/calib_libs/tinyxml2.h"

using namespace cv;
using namespace std;

/**
 * \brief create images for camera calibration:circle image for fisheye cameras
 * \param width 
 * \param height 
 * \param numX 
 * \param numY 
 */
void createCalibPicCircle(int width, int height, int numX, int numY)
{
	cv::Mat image = cv::Mat::zeros(height, width, CV_8UC1);

	int interval = min(width / (numX+1), height / (numY+1));
	int radius = interval / 4;

	numY = height / interval;
	numX = width / interval;

	for (int j = 1; j < numY; j++)
	{
		for (int i = 1; i < numX; i++)
		{
			circle(image, Point(i * interval, j * interval), radius, 255, -1);
		}
	}
	imwrite("calibCircle.jpg", image);
}

/**
 * \brief create images for camera calibration:square image for general cameras
 * \param width 
 * \param height 
 * \param numX 
 * \param numY 
 */
void createCalibPicSquare(int width, int height, int numX, int numY)
{
	int sideLen = min(width / numX, height / numY);

	numX = width / sideLen;
	numY = height / sideLen;

	width = sideLen * numX;
	height = sideLen * numY;

	cv::Mat whiteImg(sideLen, sideLen, CV_8UC1, 255);
	cv::Mat image = cv::Mat::zeros(height, width, CV_8UC1);

	for (int j = 0; j < numY; j++, j++)
	{
		for (int i = 0; i < numX; i++, i++)
		{
			whiteImg.copyTo(image(Rect(i * sideLen, j * sideLen, sideLen, sideLen)));
		}
	}

	for (int j = 1; j < numY; j++, j++)
	{
		for (int i = 1; i < numX; i++, i++)
		{
			whiteImg.copyTo(image(Rect(i * sideLen, j * sideLen, sideLen, sideLen)));
		}
	}

	imwrite("calibSquare.jpg", image);

}

void createStripePic_withSquare(int width, int height, int numX, int numY, int slope)
{
	int sideLen = min(width / numX, height / numY);

	numX = width / sideLen;
	numY = height / sideLen;

	width = sideLen * numX;
	height = sideLen * numY;

	cv::Mat whiteImg(sideLen, sideLen, CV_8UC1, 255);
	cv::Mat imageStripeH1 = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat imageStripeH2 = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat imageStripeV1 = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat imageStripeV2 = cv::Mat::zeros(height, width, CV_8UC1);

	// Make vertical pattern
	for (int j = 0; j < numY; j++)
	{
		for (int i = 0; i < numX; i++,i++)
		{
			whiteImg.copyTo(imageStripeV1(Rect(i * sideLen, j * sideLen, sideLen, sideLen)));
		}
	}
	imageStripeV2 = ~imageStripeV1;

	cv::blur(imageStripeV1, imageStripeV1, cv::Size2i(slope + 1, 1));
	cv::blur(imageStripeV2, imageStripeV2, cv::Size2i(slope + 1, 1));
	imwrite("pattern0_square.jpg", imageStripeV1);
	imwrite("pattern1_square.jpg", imageStripeV2);


	// Make horizontal pattern
	for (int j = 0; j < numY; j++,j++)
	{
		for (int i = 0; i < numX; i++)
		{
			whiteImg.copyTo(imageStripeH1(Rect(i * sideLen, j * sideLen, sideLen, sideLen)));
		}
	}
	imageStripeH2 = ~imageStripeH1;

	cv::blur(imageStripeH1, imageStripeH1, cv::Size2i(1,slope + 1));
	cv::blur(imageStripeH2, imageStripeH2, cv::Size2i(1, slope + 1));

	imwrite("pattern2_square.jpg", imageStripeH1);
	imwrite("pattern3_square.jpg", imageStripeH2);
}

/**
 * \brief open the target camera and capture images for calibration
 *			\notice that the location of the camera is supposed to be fixed during the capturing
 */
void myCameraCalibration(std::string cameraParaPath)
{
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		std::cout << "Camera open failed!" << std::endl;
		return;
	}
	cap.set(CAP_PROP_FOURCC, 'GPJM');
	cv::Size imgSize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));

	cv::Size patternSize(5, 7);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;


	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec;		//store the detected inner corners of each image
	cv::Mat img;
	cv::Mat imgGrey;
	while (cornerPtsVec.size() < 20)
	{
		cap.read(img);
		cvtColor(img, imgGrey, COLOR_BGR2GRAY);

		std::vector<Point2f> cornerPts;
		bool patternFound = findChessboardCorners(imgGrey, patternSize, cornerPts, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
			+ CALIB_CB_FAST_CHECK);
		if (patternFound)
		{
			cornerSubPix(imgGrey, cornerPts, cv::Size(11, 11), cv::Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
			cornerPtsVec.push_back(cornerPts);
			drawChessboardCorners(imgGrey, patternSize, cornerPts, patternFound);
			cornerPtsVec.push_back(img);
		}
	}
	cap.release();


	//camera calibration
	cv::Size2f squareSize(35.0, 36.2);		//the real size of each grid in the chess board,which is measured manually by ruler

	cv::Mat K = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));	//the inner parameters of camera
	cv::Mat D = cv::Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> vectorMatT;									//matrix T of each image
	std::vector<cv::Mat> vectorMatR;									//matrix R of each image

	std::vector<Point3f> tempPts;
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3f realPt;
			realPt.x = i * squareSize.width;
			realPt.y = j * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}
	std::vector<std::vector<Point3f> > objPts3d;					 	//calculated coordination of corners in world coordinate system
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		objPts3d.push_back(tempPts);
	}

	calibrateCamera(objPts3d, cornerPtsVec, imgSize, 
		K, D, vectorMatR, vectorMatT, 0);


	//evaluate the result of the camera calibration,calculate the error of calibration in each image
	double totalErr = 0.0;
	double err = 0.0;
	std::vector<Point2f> imgPts_2d;		//store the rechecked points' coordination
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		std::vector<Point3f> tempPts = objPts3d[i];  //the actual coordination of point in 3d corrdinate system
		projectPoints(tempPts, vectorMatR[i], vectorMatT[i], K, D, imgPts_2d);

		//calculate the error
		std::vector<Point2f> tempImagePoint = cornerPtsVec[i]; //the detected corner coordination in the image
		cv::Mat tempImgPt = cv::Mat(1, tempImagePoint.size(), CV_32FC2);
		cv::Mat recheckImgPt = cv::Mat(1, imgPts_2d.size(), CV_32FC2);

		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			recheckImgPt.at<cv::Vec2f>(0, j) = cv::Vec2f(imgPts_2d[j].x, imgPts_2d[j].y);
			tempImgPt.at<cv::Vec2f>(0, j) = cv::Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(recheckImgPt, tempImgPt, NORM_L2);
		totalErr += err / gridPatternNum;
	}

	std::cout << "总平均误差：" << totalErr / cornerPtsVec.size() << std::endl;


	//output the calibration result
	std::cout << "相机内参数矩阵：\n" << K << std::endl;
	std::cout << "相机畸变参数[k1,k2,k3,p1,p2]:\n" << D << std::endl;
	cv::Mat rotationMatrix = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		Rodrigues(vectorMatR[i], rotationMatrix);
		std::cout << "第" << i + 1 << "幅图像的旋转矩阵：\n" << rotationMatrix << std::endl << std::endl;
		std::cout << "第" << i + 1 << "幅图像的平移矩阵：\n" << vectorMatT[i] << std::endl << std::endl << std::endl;
	}

	//store the calibration result to the .xml file
	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "CameraInnerPara" << K;
	fn << "CameraDistPara" << D;
	fn.release();
}

/**
 * \brief using pre-captured image to calibrate the camera which is used to capture these image
 *			\notice that the location of the camera is supposed to be fixed during the pre-capturing
 * \param imgFilePath :the path of the folder to store the images
 */
void myCameraCalibration(std::string imgFilePath, std::string cameraParaPath)
{
	//load all the images in the folder
	String filePath = imgFilePath + "/*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	cv::Size patternSize(9, 6);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = 54;

	cv::Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec;		//store the detected inner corners of each image
	for (int i = 0; i < fileNames.size(); i++)
	{
		cv::Mat img = imread(fileNames[i], IMREAD_GRAYSCALE);
		if (i == 0)
		{
			imgSize.width = img.cols;
			imgSize.height = img.rows;
		}

		std::vector<Point2f> cornerPts;
		bool patternFound = findChessboardCorners(img, patternSize, cornerPts, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
			+ CALIB_CB_FAST_CHECK);
		if (patternFound)
		{
			cornerSubPix(img, cornerPts, cv::Size(11, 11), cv::Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
			cornerPtsVec.push_back(cornerPts);
			drawChessboardCorners(img, patternSize, cornerPts, patternFound);
		}
	}


	//camera calibration
	cv::Size2f squareSize(35.0, 36.2);		//the real size of each grid in the chess board,which is measured manually by ruler
	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	cv::Mat K = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));	//the inner parameters of camera
	cv::Mat D = cv::Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> vectorMatT;									//matrix T of each image
	std::vector<cv::Mat> vectorMatR;									//matrix R of each image

	for (std::vector<std::vector<Point2f>>::iterator itor = cornerPtsVec.begin(); itor != cornerPtsVec.end(); itor++)
	{
		std::vector<Point3f> tempPts;
		for (int i = 0; i < patternSize.height; i++)
		{
			for (int j = 0; j < patternSize.width; j++)
			{
				Point3f realPt;
				realPt.x = i * squareSize.width;
				realPt.y = j * squareSize.height;
				realPt.z = 0;
				tempPts.push_back(realPt);
			}
		}
		objPts3d.push_back(tempPts);
	}

	calibrateCamera(objPts3d, cornerPtsVec, imgSize, K, D, vectorMatR, vectorMatT, 0);


	//evaluate the result of the camera calibration,calculate the error of calibration in each image
	double totalErr = 0.0;
	double err = 0.0;
	std::vector<Point2f> imgPts_2d;		//store the rechecked points' coordination
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		std::vector<Point3f> tempPts = objPts3d[i];  //the actual coordination of point in 3d corrdinate system
		projectPoints(tempPts, vectorMatR[i], vectorMatT[i], K, D, imgPts_2d);

		//calculate the error
		std::vector<Point2f> tempImagePoint = cornerPtsVec[i]; //the detected corner coordination in the image
		cv::Mat tempImgPt = cv::Mat(1, tempImagePoint.size(), CV_32FC2);
		cv::Mat recheckImgPt = cv::Mat(1, imgPts_2d.size(), CV_32FC2);

		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			recheckImgPt.at<cv::Vec2f>(0, j) = cv::Vec2f(imgPts_2d[j].x, imgPts_2d[j].y);
			tempImgPt.at<cv::Vec2f>(0, j) = cv::Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(recheckImgPt, tempImgPt, NORM_L2);
		totalErr += err / gridPatternNum;
	}

	std::cout << "总平均误差：" << totalErr / cornerPtsVec.size() << std::endl;


	//output the calibration result
	std::cout << "相机内参数矩阵：\n" << K << std::endl;
	std::cout << "相机畸变参数[k1,k2,k3,p1,p2]:\n" << D << std::endl;
	cv::Mat rotationMatrix = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		Rodrigues(vectorMatR[i], rotationMatrix);
		std::cout << "第" << i + 1 << "幅图像的旋转矩阵：\n" << rotationMatrix << std::endl << std::endl;
		std::cout << "第" << i + 1 << "幅图像的平移矩阵：\n" << vectorMatT[i] << std::endl << std::endl << std::endl;
	}

	//store the calibration result to the .xml file
	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "CameraInnerPara" << K;
	fn << "CameraDistPara" << D;
	fn.release();
}

/**
 * \brief using pre-calibrated camera inner parameters and distort parameters to undistort the images captured by the camera
 * \param cameraParaPath :the .xml file path of pre-calculated parameters of camera
 */
void myCameraUndistort(std::string cameraParaPath)
{
	cv::Mat K;
	cv::Mat D;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["CameraInnerPara"] >> K;
	fn["CameraDistPara"] >> D;
	fn.release();


	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		std::cout << "Camera open failed!" << std::endl;
		return;
	}
	cap.set(CAP_PROP_FOURCC, 'GPJM');

	while (1)
	{
		cv::Mat img;
		cv::Mat undistortImg;
		cap.read(img);
		imshow("originView", img);
		undistort(img, undistortImg, K, D);
		imshow("undistortView", undistortImg);
		waitKey(10);
	}
	cap.release();
}

/**
 * \brief using pre-calibrated camera inner parameters and distort parameters to undistort the images captured by the camera
 * \param imgFilePath :the file path of images to be undistored,which is captured by the particular camera
 * \param cameraParaPath :the .xml file path of pre-calculated parameters of camera
 */
void myCameraUndistort(std::string imgFilePath, std::string cameraParaPath)
{
	cv::Mat K;
	cv::Mat D;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["CameraInnerPara"] >> K;
	fn["CameraDistPara"] >> D;
	fn.release();

	String filePath = imgFilePath + "/*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	cv::Mat undistortImg;
	std::vector<cv::Mat> undistortImgs;
	for (int i = 0; i < fileNames.size(); i++)
	{
		cv::Mat img = imread(fileNames[i]);
		undistort(img, undistortImg, K, D);
		imwrite(imgFilePath + "/" + std::to_string(i) + ".jpg", undistortImg);
		undistortImgs.push_back(undistortImg);
	}
}

/**
 * \brief stereo calibration by opening camera and capturing the chess board images
 * \param cameraParaPath
 */
double stereoCamCalibration(std::string cameraParaPath)
{
	//openning the two cameras:left camera, right camera
	std::string url_left = "rtsp://admin:yanfa1304@192.168.43.6:554/80";
	std::string url_right = "rtsp://admin:yanfa1304@192.168.43.178:554/80";

	VideoCapture cap_left(url_left);//如果是笔记本，0打开的是自带的摄像头，1 打开外接的相机
	VideoCapture cap_right(url_right);
	//double rate = capture.get(CV_CAP_PROP_FPS);//视频的帧率

	if (!cap_left.isOpened() || !cap_right.isOpened())
	{
		std::cout << "Left or right camera open failed!" << std::endl;
		return -1;
	}
	cap_left.set(CAP_PROP_FOURCC, 'GPJM');
	cap_left.set(CAP_PROP_FRAME_HEIGHT, IMG_CAPTURE_HEIGHT);		//rows
	cap_left.set(CAP_PROP_FRAME_WIDTH, IMG_CAPTURE_WIDTH);//col

	cap_right.set(CAP_PROP_FOURCC, 'GPJM');
	cap_right.set(CAP_PROP_FRAME_HEIGHT, IMG_CAPTURE_HEIGHT);
	cap_right.set(CAP_PROP_FRAME_WIDTH, IMG_CAPTURE_WIDTH);

	cv::Size imgSize(IMG_CAPTURE_WIDTH, IMG_CAPTURE_HEIGHT);


	cv::Size patternSize(9, 6);		//5:the number of inner corners in each row of the chess board
												//7:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;


	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f> > cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	while (cornerPtsVec_left.size() < 20)
	{
		cv::Mat img_left, img_right;
		cv::Mat imgGrey_left, imgGrey_right;
		cap_left.read(img_left);
		cvtColor(img_left, imgGrey_left, COLOR_BGR2GRAY);
		cap_right.read(img_right);
		cvtColor(img_right, imgGrey_right, COLOR_BGR2GRAY);

		std::vector<Point2f> cornerPts_left;
		std::vector<Point2f> cornerPts_right;

		bool patternFound_left = findChessboardCorners(imgGrey_left, patternSize, cornerPts_left, 
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		bool patternFound_right = findChessboardCorners(imgGrey_right, patternSize, cornerPts_right, 
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

		if (patternFound_left && patternFound_right)
		{
			cornerSubPix(imgGrey_left, cornerPts_left, cv::Size(11, 11), cv::Size(-1, -1), 
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_left.push_back(cornerPts_left);
			drawChessboardCorners(imgGrey_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(imgGrey_right, cornerPts_right, cv::Size(11, 11), cv::Size(-1, -1), 
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_right.push_back(cornerPts_right);
			drawChessboardCorners(imgGrey_right, patternSize, cornerPts_right, patternFound_right);
		}
	}
	cap_left.release();
	cap_right.release();

	//stereo calibration
	cv::Size2f squareSize(100, 100);				//the real size of each grid in the chess board,which is measured manually by ruler,单位：mm

	//get the coordinations of cross points in the real world:stored along rows
	std::vector<Point3f> tempPts;
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3f realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);	
		}
	}
	std::vector<std::vector<Point3f> > objPts3d;			//calculated coordination of corners in world coordinate system
	for (int i = 0; i < cornerPtsVec_left.size(); i++)
	{
		objPts3d.push_back(tempPts);
	}

	//single camera calibration flag
	int flag = 0;
	flag |= CALIB_FIX_PRINCIPAL_POINT;
	flag |= CALIB_FIX_ASPECT_RATIO;
	flag |= CALIB_ZERO_TANGENT_DIST;
	flag |= CALIB_RATIONAL_MODEL;
	flag |= CALIB_FIX_K3;
	flag |= CALIB_FIX_K4;
	flag |= CALIB_FIX_K5;

	//calibrate left camera
	cv::Mat K_left = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));		//the inner parameters of camera
	cv::Mat D_left = cv::Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_left;										//matrix T of each image
	std::vector<cv::Mat> R_left;										//matrix R of each image
	double rmsLeft = calibrateCamera(objPts3d, cornerPtsVec_left, imgSize,
		K_left, D_left, 
		R_left, T_left, flag);

	//calibrate right camera
	cv::Mat K_right = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));	//the inner parameters of camera
	cv::Mat D_right = cv::Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_right;									//matrix T of each image
	std::vector<cv::Mat> R_right;									//matrix R of each image
	double rmsRight = calibrateCamera(objPts3d, cornerPtsVec_right, imgSize,
		K_right, D_right, 
		R_right, T_right, flag);


	//stereo calibration
	cv::Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	cv::Mat matrixE;				//essential matrix E
	cv::Mat matrixF;				//fundamental matrix F
	int stereoFlag = 0;
	stereoFlag |= CALIB_FIX_INTRINSIC;
	stereoFlag |= CALIB_SAME_FOCAL_LENGTH;

	double rms = stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		K_left, D_left,
		K_right, D_right,
		imgSize, matrixR, matrixT, matrixE, matrixF, stereoFlag,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "Imgcv::Size" << imgSize;
	fn << "Left_CameraInnerPara" << K_left;
	fn << "Left_CameraDistPara" << D_left;
	fn << "Right_CameraInnerPara" << K_right;
	fn << "Right_CameraDistPara" << D_right;
	fn << "R2L_Rotation_cv::Matrix" << matrixR;
	fn << "R2L_Translate_cv::Matrix" << matrixT;
	fn << "Essential_cv::Matrix" << matrixE;
	fn << "Fundamental_cv::Matrix" << matrixF;
	fn.release();

	return rms;

	double err = 0;
	int npoints = 0;
	std::vector<Vec3f> lines[2];
	for (int i = 0; i < cornerPtsVec_left.size(); i++)
	{
		int npt = (int)cornerPtsVec_left[i].size();
		cv::Mat imgpt[2];
		imgpt[0] = cv::Mat(cornerPtsVec_left[i]);
		undistortPoints(imgpt[0], imgpt[0], K_left, D_left, cv::Mat(), K_left);
		computeCorrespondEpilines(imgpt[0], 0 + 1, matrixF, lines[0]);

		imgpt[1] = cv::Mat(cornerPtsVec_right[i]);
		undistortPoints(imgpt[1], imgpt[1], K_right, D_right, cv::Mat(), K_right);
		computeCorrespondEpilines(imgpt[1], 1 + 1, matrixF, lines[1]);

		for (int j = 0; j < npt; j++)
		{
			double errij = fabs(cornerPtsVec_left[i][j].x*lines[1][j][0] +
				cornerPtsVec_left[i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(cornerPtsVec_right[i][j].x*lines[0][j][0] +
					cornerPtsVec_right[i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	std::cout << "average epipolar err = " << err / npoints << std::endl;
}

/**
 * \brief detect chess board corners for \stereo_calibration(\findChessboardCorners())
 * \param imgFilePathL 
 * \param ptsL :pts for left camera
 * \param ptsR :pts for right camera
 * \param ptsReal :pts in the real world coordination
 * \param corRowNum :number of corners each row in the chessboard
 * \param corColNum :number of corners each col in the chessboard
 * \return 
 */
bool ptsCalib(std::string imgFilePathL, cv::Size& imgSize, 
	douVecPt2f& ptsL, douVecPt2f& ptsR, douVecPt3f& ptsReal,
	int corRowNum, int corColNum)
{
	if(!ptsL.empty())
	{
		ptsL.clear();
	}

	if(!ptsR.empty())
	{
		ptsR.clear();
	}

	if(!ptsReal.empty())
	{
		ptsReal.clear();
	}

	//read the two cameras' pre-captured  images:left camera, right camera
	//load all the images in the folder
	String filePath = imgFilePathL + "/*L.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	cv::Size patternSize(corRowNum, corColNum);		//0:the number of inner corners in each row of the chess board
																//1:the number of inner corners in each col of the chess board
	//detect the inner corner in each chess image
	int x_expand_half = 0;
	int y_expand_half = 0;
	for (int i = 0; i < fileNames.size(); i++)
	{
		cv::Mat img_left = imread(fileNames[i]);
		cv::Mat img_right = imread(fileNames[i].substr(0, fileNames[i].length() - 5) + "R.jpg");

		if (img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "img reading error" << std::endl;
			return false;
		}

		if (i == 0)
		{
			imgSize.width = img_left.cols + x_expand_half * 2;
			imgSize.height = img_left.rows + y_expand_half * 2;
		}

		cv::Mat imgL_border, imgR_border;
		copyMakeBorder(img_left, imgL_border, y_expand_half, y_expand_half, x_expand_half, x_expand_half, BORDER_CONSTANT);
		copyMakeBorder(img_right, imgR_border, y_expand_half, y_expand_half, x_expand_half, x_expand_half, BORDER_CONSTANT);

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(imgL_border, patternSize, cornerPts_left, 
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		bool patternFound_right = findChessboardCorners(imgR_border, patternSize, cornerPts_right, 
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		if (patternFound_left && patternFound_right)
		{
			cv::Mat imgL_gray, imgR_gray;
			cvtColor(imgL_border, imgL_gray, COLOR_RGB2GRAY);
			cornerSubPix(imgL_gray, cornerPts_left, cv::Size(3, 3), cv::Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
			ptsL.push_back(cornerPts_left);
			drawChessboardCorners(imgL_border, patternSize, cornerPts_left, patternFound_left);

			cvtColor(imgR_border, imgR_gray, COLOR_RGB2GRAY);
			cornerSubPix(imgR_gray, cornerPts_right, cv::Size(3, 3), cv::Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
			ptsR.push_back(cornerPts_right);
			drawChessboardCorners(imgR_border, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	cv::Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<Point3f> tempPts;
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3f realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}
	//for (int i = patternSize.height - 1; i >= 0; i--)
	//{
	//	for (int j = patternSize.width - 1; j >= 0; j--)
	//	{
	//		Point3f realPt;
	//		realPt.x = j * squareSize.width;
	//		realPt.y = i * squareSize.height;
	//		realPt.z = 0;
	//		tempPts.push_back(realPt);
	//	}
	//}

	for (int i = 0; i < ptsL.size(); i++)
	{
		ptsReal.push_back(tempPts);
	}

	return true;
}

/**
 * \brief detect chessboard cross points in the undistorted images for stereo calibration
 * \param imgs 
 * \param ptsL 
 * \param ptsR 
 * \param ptsReal 
 * \param corRowNum 
 * \param corColNum 
 * \return 
 */
bool ptsCalib(std::vector<cv::Mat> imgsL, std::vector<cv::Mat> imgsR, 
	douVecPt2f& ptsL, douVecPt2f& ptsR, douVecPt3f& ptsReal, int corRowNum, int corColNum)
{
	if(imgsL.size() != imgsR.size() || imgsL.size() <= 0)
	{
		cout << "images error" << endl;
		return false;
	}

	if (!ptsL.empty())
	{
		ptsL.clear();
	}

	if (!ptsR.empty())
	{
		ptsR.clear();
	}

	if (!ptsReal.empty())
	{
		ptsReal.clear();
	}

	//
	cv::Size patternSize(corRowNum, corColNum);		//0:the number of inner corners in each row of the chess board
																//1:the number of inner corners in each col of the chess board

	//detect the inner corner in each chess image
	for (int i = 0; i < imgsL.size(); i++)
	{
		cv::Mat img_left = imgsL[i];
		cv::Mat img_right = imgsR[i];

		if (img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "image error" << std::endl;
			return false;
		}

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(img_left, patternSize, cornerPts_left,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		bool patternFound_right = findChessboardCorners(img_right, patternSize, cornerPts_right,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		if (patternFound_left && patternFound_right)
		{
			cv::Mat imgL_gray, imgR_gray;
			cvtColor(img_left, imgL_gray, COLOR_RGB2GRAY);
			cornerSubPix(imgL_gray, cornerPts_left, cv::Size(3, 3), cv::Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
			ptsL.push_back(cornerPts_left);
			drawChessboardCorners(img_left, patternSize, cornerPts_left, patternFound_left);

			cvtColor(img_right, imgR_gray, COLOR_RGB2GRAY);
			cornerSubPix(imgR_gray, cornerPts_right, cv::Size(3, 3), cv::Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
			ptsR.push_back(cornerPts_right);
			drawChessboardCorners(img_right, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	cv::Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<Point3f> tempPts;
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3f realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}
	for (int i = 0; i < ptsL.size(); i++)
	{
		ptsReal.push_back(tempPts);
	}

	return true;
}


/**
 * \brief detect chessboard cross points in distorted images for single calibration
 * \param imgFilePath 
 * \param imgSize 
 * \param pts 
 * \param ptsReal 
 * \param corRowNum 
 * \param corColNum 
 * \return 
 */
bool ptsCalib_Single(std::string imgFilePath, cv::Size& imgSize, 
	douVecPt2f& pts, douVecPt3f& ptsReal, int corRowNum, int corColNum)
{
	if (!pts.empty())
	{
		pts.clear();
	}

	if (!ptsReal.empty())
	{
		ptsReal.clear();
	}

	//read the two cameras' pre-captured  images:left camera, right camera
	//load all the images in the folder
	String filePath = imgFilePath + "/*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	cv::Size patternSize(corRowNum, corColNum);		//0:the number of inner corners in each row of the chess board
												//1:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;

	//detect the inner corner in each chess image
	int x_expand_half = 0;
	int y_expand_half = 0;
	for (int i = 0; i < fileNames.size(); i++)
	{
		cv::Mat img = imread(fileNames[i]);

		if (i == 0)
		{
			imgSize.width = img.cols + x_expand_half * 2;
			imgSize.height = img.rows + y_expand_half * 2;
		}

		cv::Mat imgL_border;
		copyMakeBorder(img, imgL_border, y_expand_half, y_expand_half, x_expand_half, x_expand_half, BORDER_CONSTANT, 0);
		//imwrite(fileNames[i].substr(0, fileNames[i].size() - 4) + "_border.jpg", imgL_border);

		std::vector<Point2f> cornerPts;
		bool patternFound = findChessboardCorners(imgL_border, patternSize, cornerPts,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		if (patternFound)
		{
			cv::Mat img_gray;
			cvtColor(imgL_border, img_gray, COLOR_RGB2GRAY);
			cornerSubPix(img_gray, cornerPts, cv::Size(3, 3), cv::Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
			pts.push_back(cornerPts);
			drawChessboardCorners(imgL_border, patternSize, cornerPts, patternFound);
			cout << fileNames[i] << "\taccept\n";
		}
		else
		{
			cout << fileNames[i] << "\treject\n";
		}
	}

	//two cameras calibration
	cv::Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<Point3f> tempPts;
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3f realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}
	//for (int i = patternSize.height - 1; i >= 0; i--)
	//{
	//	for (int j = patternSize.width - 1; j >= 0; j--)
	//	{
	//		Point3f realPt;
	//		realPt.x = j * squareSize.width;
	//		realPt.y = i * squareSize.height;
	//		realPt.z = 0;
	//		tempPts.push_back(realPt);
	//	}
	//}

	for (int i = 0; i < pts.size(); i++)
	{
		ptsReal.push_back(tempPts);
	}

	return true;
}

/**
 * \brief detect chessboard cross points in the undistorted images
 * \param imgs 
 * \param ptsL 
 * \param ptsR 
 * \param ptsReal 
 * \param corRowNum 
 * \param corColNum 
 * \return 
 */
bool ptsCalib_Single(std::vector<cv::Mat> imgs, douVecPt2f& pts, douVecPt3f& ptsReal, int corRowNum,
	int corColNum)
{
	if(imgs.empty())
	{
		return false;
	}

	if (!pts.empty())
	{
		pts.clear();
	}

	if (!ptsReal.empty())
	{
		ptsReal.clear();
	}

	cv::Size patternSize(corRowNum, corColNum);		//0:the number of inner corners in each row of the chess board
																//1:the number of inner corners in each col of the chess board

	//detect the inner corner in each chess image
	for (int i = 0; i < imgs.size(); i++)
	{
		cv::Mat img = imgs[i];

		std::vector<Point2f> cornerPts;
		bool patternFound = findChessboardCorners(img, patternSize, cornerPts,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		if (patternFound)
		{
			cv::Mat img_gray;
			cvtColor(img, img_gray, COLOR_RGB2GRAY);
			cornerSubPix(img_gray, cornerPts, cv::Size(3, 3), cv::Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
			pts.push_back(cornerPts);
			drawChessboardCorners(img, patternSize, cornerPts, patternFound);
		}
	}

	//two cameras calibration
	cv::Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<Point3f> tempPts;
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3f realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}
	for (int i = 0; i < pts.size(); i++)
	{
		ptsReal.push_back(tempPts);
	}

	return true;
}

void createMask_lines(cv::Mat& dst)
{
	vector<vector<cv::Point2i> > contours;
	{
		vector<cv::Point2i> oneContour;

		cv::Point2i p1(2559, 0);
		cv::Point2i p2(2061, 0);
		cv::Point2i p3(2070, 12);
		cv::Point2i p4(2086, 31);
		cv::Point2i p5(2116, 61);
		cv::Point2i p6(2141, 88);
		cv::Point2i p7(2179, 130);
		cv::Point2i p8(2246, 217);
		cv::Point2i p9(2314, 324);
		cv::Point2i p10(2329, 345);
		cv::Point2i p11(2346, 362);
		cv::Point2i p12(2397, 391);
		cv::Point2i p13(2425, 499);
		cv::Point2i p14(2418, 510);
		cv::Point2i p15(2424, 577);
		cv::Point2i p16(2414, 599);
		cv::Point2i p17(2400, 666);
		cv::Point2i p18(2399, 712);
		cv::Point2i p19(2400, 742);
		cv::Point2i p20(2401, 768);
		cv::Point2i p21(2402, 779);
		cv::Point2i p22(2404, 800);
		cv::Point2i p23(2407, 815);
		cv::Point2i p24(2409, 834);
		cv::Point2i p25(2416, 862);
		cv::Point2i p26(2425, 885);
		cv::Point2i p27(2425, 891);
		cv::Point2i p28(2418, 935);
		cv::Point2i p29(2424, 962);
		cv::Point2i p30(2403, 1050);
		cv::Point2i p31(2302, 1303);
		cv::Point2i p32(2221, 1439);
		cv::Point2i p33(2559, 1439);

		oneContour.push_back(p1);
		oneContour.push_back(p2);
		oneContour.push_back(p3);
		oneContour.push_back(p4);
		oneContour.push_back(p5);
		oneContour.push_back(p6);
		oneContour.push_back(p7);
		oneContour.push_back(p8);
		oneContour.push_back(p9);
		oneContour.push_back(p10);
		oneContour.push_back(p11);
		oneContour.push_back(p12);
		oneContour.push_back(p13);
		oneContour.push_back(p14);
		oneContour.push_back(p15);
		oneContour.push_back(p16);
		oneContour.push_back(p17);
		oneContour.push_back(p18);
		oneContour.push_back(p19);
		oneContour.push_back(p20);
		oneContour.push_back(p21);
		oneContour.push_back(p22);
		oneContour.push_back(p23);
		oneContour.push_back(p24);
		oneContour.push_back(p25);
		oneContour.push_back(p26);
		oneContour.push_back(p27);
		oneContour.push_back(p28);
		oneContour.push_back(p29);
		oneContour.push_back(p30);
		oneContour.push_back(p31);
		oneContour.push_back(p32);
		oneContour.push_back(p33);

		contours.push_back(oneContour);
	}
	//
	{
		vector<cv::Point2i> oneContour;

		cv::Point2i p1(0, 0);
		cv::Point2i p2(357, 0);
		cv::Point2i p3(274, 116);
		cv::Point2i p4(241, 179);
		cv::Point2i p5(191, 284);
		cv::Point2i p6(158, 372);
		cv::Point2i p7(130, 480);
		cv::Point2i p8(113, 590);
		cv::Point2i p9(107, 685);
		cv::Point2i p10(110, 801);
		cv::Point2i p11(122, 903);
		cv::Point2i p12(149, 1024);
		cv::Point2i p13(188, 1129);
		cv::Point2i p14(285, 1356);
		cv::Point2i p15(350, 1439);
		cv::Point2i p16(0, 1439);

		oneContour.push_back(p1);
		oneContour.push_back(p2);
		oneContour.push_back(p3);
		oneContour.push_back(p4);
		oneContour.push_back(p5);
		oneContour.push_back(p6);
		oneContour.push_back(p7);
		oneContour.push_back(p8);
		oneContour.push_back(p9);
		oneContour.push_back(p10);
		oneContour.push_back(p11);
		oneContour.push_back(p12);
		oneContour.push_back(p13);
		oneContour.push_back(p14);
		oneContour.push_back(p15);
		oneContour.push_back(p16);

		contours.push_back(oneContour);
	}

	int width = 2560;
	int height = 1440;
	cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);

	drawContours(img, contours, -1, 255, FILLED);
	//imwrite("img_.jpg", img);
	bitwise_not(img, dst);
}

cv::Mat detectLines_(cv::Mat& src1, cv::Mat& src2, bool isHorizon)
{
	// Check type of img1 and img2
	if (src1.type() != CV_64FC1) {
		cv::Mat tmp;
		src1.convertTo(tmp, CV_64FC1);
		src1 = tmp;
	}
	if (src2.type() != CV_64FC1) {
		cv::Mat tmp;
		src2.convertTo(tmp, CV_64FC1);
		src2 = tmp;
	}
	cv::Mat diff = src1 - src2;
	cv::Mat cross = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
	cv::Mat cross_inv = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
	double thresh = 100;//185 for 20191017
	bool positive; // Whether previous found cross point was positive
	bool search; // Whether serching
	bool found_first;
	int val_now, val_prev;

	if (isHorizon)
	{
		// search for y direction
		for (int x = 0; x < diff.cols; x++) {
			val_prev = diff.at<double>(0, x);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int y = 1; y < diff.rows; ++y) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						if (cross.at<uchar>(y, x) != 255) {
							cross.at<uchar>(y, x) = 255;
						}
					}
					else {
						cross.at<uchar>(y - 1, x) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

		// search for inversed y direction
		for (int x = 0; x < diff.cols; x++) {
			val_prev = diff.at<double>(diff.rows - 1, x);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int y = diff.rows - 2; y > 0; --y) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						if (cross_inv.at<uchar>(y, x) != 255) {
							cross_inv.at<uchar>(y, x) = 255;
						}
					}
					else {
						cross_inv.at<uchar>(y + 1, x) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}
	}
	else
	{
		// search for x direction
		for (int y = 0; y < diff.rows; y++) {
			val_prev = diff.at<double>(y, 0);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int x = 1; x < diff.cols; ++x) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						cross.at<uchar>(y, x) = 255;
					}
					else {
						cross.at<uchar>(y, x - 1) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

		// search for inversed x direction
		for (int y = 0; y < diff.rows; y++) {
			val_prev = diff.at<double>(y, diff.cols - 1);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int x = diff.cols - 2; x > 0; --x) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						cross_inv.at<uchar>(y, x) = 255;
					}
					else {
						cross_inv.at<uchar>(y, x + 1) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}
	}

	cv::Mat dst_cross;
	cv::bitwise_and(cross, cross_inv, dst_cross);
	return dst_cross;
}

/**
 * \brief detecting lines for further work///methods following the function detectLine() in LineDetection.cpp
 * \param src1 
 * \param src2 
 * \param dst :image type is CV_8UC1
 * \param isHorizon
 */
void detectLines_(cv::Mat src1, cv::Mat src2, cv::Mat& dst, cv::Mat& dst_inv, bool isHorizon)
{
	// Check type of img1 and img2
	if (src1.type() != CV_64FC1) {
		cv::Mat tmp;
		src1.convertTo(tmp, CV_64FC1);
		src1 = tmp;
	}
	if (src2.type() != CV_64FC1) {
		cv::Mat tmp;
		src2.convertTo(tmp, CV_64FC1);
		src2 = tmp;
	}
	cv::Mat diff = src1 - src2;
	cv::Mat cross = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
	cv::Mat cross_inv = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
	double thresh = 50;//185 for 20191017
	bool positive; // Whether previous found cross point was positive
	bool search; // Whether serching
	bool found_first;
	int val_now, val_prev;

	if (isHorizon)
	{
		// search for y direction
		for (int x = 0; x < diff.cols; x++) {
			val_prev = diff.at<double>(0, x);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int y = 1; y < diff.rows; ++y) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						if (cross.at<uchar>(y, x) != 255) {
							cross.at<uchar>(y, x) = 255;
						}
					}
					else {
						cross.at<uchar>(y - 1, x) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

		// search for inversed y direction
		for (int x = 0; x < diff.cols; x++) {
			val_prev = diff.at<double>(diff.rows - 1, x);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int y = diff.rows - 2; y > 0; --y) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						if (cross_inv.at<uchar>(y, x) != 255) {
							cross_inv.at<uchar>(y, x) = 255;
						}
					}
					else {
						cross_inv.at<uchar>(y + 1, x) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}
	}
	else
	{
		// search for x direction
		for (int y = 0; y < diff.rows; y++) {
			val_prev = diff.at<double>(y, 0);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int x = 1; x < diff.cols; ++x) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						cross.at<uchar>(y, x) = 255;
					}
					else {
						cross.at<uchar>(y, x - 1) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

		// search for inversed x direction
		for (int y = 0; y < diff.rows; y++) {
			val_prev = diff.at<double>(y, diff.cols - 1);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int x = diff.cols - 2; x > 0; --x) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						cross_inv.at<uchar>(y, x) = 255;
					}
					else {
						cross_inv.at<uchar>(y, x + 1) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

	}

	//cv::bitwise_and(cross, cross_inv, dst);
	dst = cross.clone();
	dst_inv = cross_inv.clone();
}

void connectEdge(cv::Mat& src, int winSize_thres, bool isHorizon)
{
	int width = src.cols;
	int height = src.rows;

	int half_winsize_thres = winSize_thres;

	if (isHorizon)
	{
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (src.at<uchar>(y, x) == 255)
				{
					if(src.at<uchar>(y - 1, x ) == 255 || src.at<uchar>(y + 1, x) == 255)
					{
						continue;
					}
					//检查8邻域
					int num_8 = 0;
					int offset_x1[2] = { -1, 1 };
					//
					int starty = 1;
					for(int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (src.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							num_8++;
					}
					while(num_8 == 0 && starty < half_winsize_thres)
					{
						offset_x1[0]--;
						starty++;
						for(int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[0] >= 0 && x + offset_x1[0] < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							{
								src.at<uchar>(y + offset_y1 / 2, x + offset_x1[0] / 2) = 255;
								if(offset_y1 / 2 <= 0 && offset_x1[0] / 2 <= 0 && starty > 2)
								{
									x = x + offset_x1[0] / 2 - 1;
									y = y + offset_y1 / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
					//
					starty = 1;
					num_8 = 0;
					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (src.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							num_8++;
					}
					while (num_8 == 0 && starty < half_winsize_thres)
					{
						offset_x1[1]++;
						starty++;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[1] >= 0 && x + offset_x1[1] < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							{
								src.at<uchar>(y + offset_y1 / 2, x + offset_x1[1] / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}

	}
	else
	{
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (src.at<uchar>(y, x) == 255)
				{
					if (src.at<uchar>(y, x - 1) == 255 || src.at<uchar>(y, x + 1) == 255)
					{
						continue;
					}

					//检查8邻域
					int num_8 = 0;
					int offset_y1[2] = { -1, 1 };
					//
					int startx = 1;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (src.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							num_8++;
					}
					while (num_8 == 0 && startx < half_winsize_thres)
					{
						offset_y1[0]--;
						startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if(!(y + offset_y1[0] >= 0 && y + offset_y1[0] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							{
								src.at<uchar>(y + offset_y1[0] / 2, x + offset_x1 / 2) = 255;
								if (offset_x1 / 2 <= 0 && offset_y1[0] / 2 <= 0 && startx > 2)
								{
									x = x + offset_x1 / 2 - 1;
									y = y + offset_y1[0] / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
					//
					startx = 1;
					num_8 = 0;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (src.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							num_8++;
					}
					while (num_8 == 0 && startx < half_winsize_thres)
					{
						offset_y1[1]++;
						startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[1] >= 0 && y + offset_y1[1] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							{
								src.at<uchar>(y + offset_y1[1] / 2, x + offset_x1 / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}

	}
}

void connectEdge_(cv::Mat& src, int winSize_thres, bool isHorizon)
{
	int width = src.cols;
	int height = src.rows;

	int half_winsize_thres = winSize_thres;

	if (isHorizon)
	{
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (src.at<uchar>(y, x) == 255)
				{
					if (src.at<uchar>(y - 1, x) == 255 || src.at<uchar>(y + 1, x) == 255)
					{
						continue;
					}
					//检查8邻域
					int num_8 = 0;
					int offset_x1[2] = { -1, 1 };
					//
					int starty = 1;
					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (src.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							num_8++;
					}
					starty++;
					while (num_8 == 0 && -offset_x1[0] < half_winsize_thres)
					{
						offset_x1[0]--;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[0] >= 0 && x + offset_x1[0] < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							{
								src.at<uchar>(y + offset_y1 / 2, x + offset_x1[0] / 2) = 255;
								if (offset_y1 / 2 <= 0 && offset_x1[0] / 2 <= 0 && starty > 2)
								{
									x = x + offset_x1[0] / 2 - 1;
									y = y + offset_y1 / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
					//
					starty = 1;
					num_8 = 0;
					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (src.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							num_8++;
					}
					starty++;
					while (num_8 == 0 && offset_x1[1] < half_winsize_thres)
					{
						offset_x1[1]++;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[1] >= 0 && x + offset_x1[1] < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							{
								src.at<uchar>(y + offset_y1 / 2, x + offset_x1[1] / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}

	}
	else
	{
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (src.at<uchar>(y, x) == 255)
				{
					if (src.at<uchar>(y, x - 1) == 255 || src.at<uchar>(y, x + 1) == 255)
					{
						continue;
					}

					//检查8邻域
					int num_8 = 0;
					int offset_y1[2] = { -1, 1 };
					//
					int startx = 1;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (src.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							num_8++;
					}
					startx++;
					while (num_8 == 0 && -offset_y1[0] < half_winsize_thres)
					{
						offset_y1[0]--;
						//startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[0] >= 0 && y + offset_y1[0] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							{
								src.at<uchar>(y + offset_y1[0] / 2, x + offset_x1 / 2) = 255;
								if (offset_x1 / 2 <= 0 && offset_y1[0] / 2 <= 0 && startx > 2)
								{
									x = x + offset_x1 / 2 - 1;
									y = y + offset_y1[0] / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
					//
					startx = 1;
					num_8 = 0;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (src.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							num_8++;
					}
					startx++;
					while (num_8 == 0 && offset_y1[1] < half_winsize_thres)
					{
						offset_y1[1]++;
						//startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[1] >= 0 && y + offset_y1[1] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							{
								src.at<uchar>(y + offset_y1[1] / 2, x + offset_x1 / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}
	}
}

void connectEdge2(cv::Mat& src, int winSize_thres, bool isHorizon)
{
	int width = src.cols;
	int height = src.rows;

	int half_winsize_thres = winSize_thres;

	if (isHorizon)
	{
		cv::Mat tmp1, tmp2;
		tmp1 = src.clone();
		tmp2 = src.clone();

		int offset_x1[2];
		for (int y = height - 3; y <= 2 ; y--)
		{
			for (int x = width - 3; x <= 2 ; x--)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (tmp1.at<uchar>(y, x) == 255)
				{
					if (tmp1.at<uchar>(y - 1, x) == 255 || tmp1.at<uchar>(y + 1, x) == 255)
					{
						continue;
					}
					//检查8邻域
					int num_8 = 0;
					offset_x1[0] = -1;
					//
					int starty = 1;
					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (tmp1.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							num_8++;
					}
					while (num_8 == 0 && starty < half_winsize_thres)
					{
						offset_x1[0]--;
						starty++;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[0] >= 0 && x + offset_x1[0] < width))
							{
								continue;
							}
							if (tmp1.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							{
								tmp1.at<uchar>(y + offset_y1 / 2, x + offset_x1[0] / 2) = 255;
								if (offset_y1 / 2 <= 0 && offset_x1[0] / 2 <= 0 && starty > 2)
								{
									x = x + offset_x1[0] / 2 - 1;
									y = y + offset_y1 / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
				}
			}
		}
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (tmp2.at<uchar>(y, x) == 255)
				{
					if (tmp2.at<uchar>(y - 1, x) == 255 || tmp2.at<uchar>(y + 1, x) == 255)
					{
						continue;
					}
					//检查8邻域
					int num_8 = 0;
					int starty = 1;
					offset_x1[1] = 1;

					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (tmp2.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							num_8++;
					}
					while (num_8 == 0 && starty < half_winsize_thres)
					{
						offset_x1[1]++;
						starty++;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[1] >= 0 && x + offset_x1[1] < width))
							{
								continue;
							}
							if (tmp2.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							{
								tmp2.at<uchar>(y + offset_y1 / 2, x + offset_x1[1] / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}

		bitwise_and(tmp1, tmp2, src);

	}
	else
	{

		cv::Mat tmp1, tmp2;
		tmp1 = src.clone();
		tmp2 = src.clone();

		int offset_y1[2];
		for (int y = height - 3; y <= 2; y--)
		{
			for (int x = width - 3; x <= 2; x--)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (tmp1.at<uchar>(y, x) == 255)
				{
					if (tmp1.at<uchar>(y, x - 1) == 255 || tmp1.at<uchar>(y, x + 1) == 255)
					{
						continue;
					}

					//检查8邻域
					int num_8 = 0;
					offset_y1[0] = -1;
					//
					int startx = 1;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (tmp1.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							num_8++;
					}
					while (num_8 == 0 && startx < half_winsize_thres)
					{
						offset_y1[0]--;
						startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[0] >= 0 && y + offset_y1[0] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (tmp1.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							{
								tmp1.at<uchar>(y + offset_y1[0] / 2, x + offset_x1 / 2) = 255;
								if (offset_x1 / 2 <= 0 && offset_y1[0] / 2 <= 0 && startx > 2)
								{
									x = x + offset_x1 / 2 - 1;
									y = y + offset_y1[0] / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
				}
			}
		}
		//
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (tmp2.at<uchar>(y, x) == 255)
				{
					if (tmp2.at<uchar>(y, x - 1) == 255 || tmp2.at<uchar>(y, x + 1) == 255)
					{
						continue;
					}

					//检查8邻域
					offset_y1[1] = 1;
					int num_8 = 0;
					int startx = 1;

					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (tmp2.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							num_8++;
					}
					while (num_8 == 0 && startx < half_winsize_thres)
					{
						offset_y1[1]++;
						startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[1] >= 0 && y + offset_y1[1] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (tmp2.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							{
								tmp2.at<uchar>(y + offset_y1[1] / 2, x + offset_x1 / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}
		bitwise_and(tmp1, tmp2, src);
	}
}

void myGetLines(cv::Mat& src, cv::Mat& tmp, cv::Point2i startPt, std::vector<cv::Point2i>& oneLine, int lenThres, bool isHorizon)
{
	if(!oneLine.empty())
	{
		oneLine.clear();
	}
	if(isHorizon)
	{
		int max_x = startPt.x;
		int min_x = startPt.x;
		stack<int> p_x, p_y;
		p_x.push(startPt.x);
		p_y.push(startPt.y);
		while (!p_x.empty())
		{
			cv::Point2i p(p_x.top(), p_y.top());
			if (p.x > max_x)
			{
				max_x = p.x;
			}
			if (p.x < min_x)
			{
				min_x = p.x;
			}
			oneLine.push_back(p);
			p_x.pop(); p_y.pop();
			if ((p.y != 0 && p.x != 0) && (tmp.at<uchar>(p.y - 1, p.x - 1) == 255)) { // Top left
				tmp.at<uchar>(p.y - 1, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y - 1);
			}
			if ((p.y != 0) && (tmp.at<uchar>(p.y - 1, p.x) == 255)) { // Top
				tmp.at<uchar>(p.y - 1, p.x) = 0;
				p_x.push(p.x); p_y.push(p.y - 1);
			}
			if ((p.y != 0 && p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y - 1, p.x + 1) == 255)) { // Top right
				tmp.at<uchar>(p.y - 1, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y - 1);
			}
			if ((p.x != 0) && (tmp.at<uchar>(p.y, p.x - 1) == 255)) { // left
				tmp.at<uchar>(p.y, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y);
			}
			if ((p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y, p.x + 1) == 255)) { // Right
				tmp.at<uchar>(p.y, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y);
			}
			if ((p.y != tmp.rows - 1 && p.x != 0) && (tmp.at<uchar>(p.y + 1, p.x - 1) == 255)) { // Down left
				tmp.at<uchar>(p.y + 1, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y + 1);
			}
			if ((p.y != tmp.rows - 1) && (tmp.at<uchar>(p.y + 1, p.x) == 255)) { // Down
				tmp.at<uchar>(p.y + 1, p.x) = 0;
				p_x.push(p.x); p_y.push(p.y + 1);
			}
			if ((p.y != tmp.rows - 1 && p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y + 1, p.x + 1) == 255)) { // Down right
				tmp.at<uchar>(p.y + 1, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y + 1);
			}
		}

		if ((max_x - min_x) < lenThres)
		{
			for (std::vector<cv::Point2i>::iterator it = oneLine.begin(); it != oneLine.end(); it++)
			{
				cv::Point2i pt = *it;
				src.at<uchar>(pt.y, pt.x) = 0;
			}
			oneLine.clear();
		}
	}
	else
	{
		int max_y = startPt.y;
		int min_y = startPt.y;
		stack<int> p_x, p_y;
		p_x.push(startPt.x);
		p_y.push(startPt.y);
		while (!p_x.empty())
		{
			cv::Point2i p(p_x.top(), p_y.top());
			if (p.y > max_y)
			{
				max_y = p.y;
			}
			if (p.y < min_y)
			{
				min_y = p.y;
			}
			oneLine.push_back(p);
			p_x.pop(); p_y.pop();
			if ((p.y != 0 && p.x != 0) && (tmp.at<uchar>(p.y - 1, p.x - 1) == 255)) { // Top left
				tmp.at<uchar>(p.y - 1, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y - 1);
			}
			if ((p.y != 0) && (tmp.at<uchar>(p.y - 1, p.x) == 255)) { // Top
				tmp.at<uchar>(p.y - 1, p.x) = 0;
				p_x.push(p.x); p_y.push(p.y - 1);
			}
			if ((p.y != 0 && p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y - 1, p.x + 1) == 255)) { // Top right
				tmp.at<uchar>(p.y - 1, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y - 1);
			}
			if ((p.x != 0) && (tmp.at<uchar>(p.y, p.x - 1) == 255)) { // left
				tmp.at<uchar>(p.y, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y);
			}
			if ((p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y, p.x + 1) == 255)) { // Right
				tmp.at<uchar>(p.y, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y);
			}
			if ((p.y != tmp.rows - 1 && p.x != 0) && (tmp.at<uchar>(p.y + 1, p.x - 1) == 255)) { // Down left
				tmp.at<uchar>(p.y + 1, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y + 1);
			}
			if ((p.y != tmp.rows - 1) && (tmp.at<uchar>(p.y + 1, p.x) == 255)) { // Down
				tmp.at<uchar>(p.y + 1, p.x) = 0;
				p_x.push(p.x); p_y.push(p.y + 1);
			}
			if ((p.y != tmp.rows - 1 && p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y + 1, p.x + 1) == 255)) { // Down right
				tmp.at<uchar>(p.y + 1, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y + 1);
			}
		}

		if ((max_y - min_y) < lenThres)
		{
			for (std::vector<cv::Point2i>::iterator it = oneLine.begin(); it != oneLine.end(); it++)
			{
				cv::Point2i pt = *it;
				src.at<uchar>(pt.y, pt.x) = 0;
			}
			oneLine.clear();
		}
	}
}

void removeShortEdges(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon, RIGHT_COUNT_SIDE mode)
{
	int width = src.cols;
	int height = src.rows;

	int count = 0;
	if(!lines.empty())
	{
		lines.clear();
	}

    cv:Mat tmp = src.clone();

	if (isHorizon)
	{
		if (mode == TOP_LEFT || mode == BOTTOM_LEFT)
		{
			for (int y = 2; y < height - 2; y++)
			{
				for (int x = 2; x < width - 2; x++)
				{
					if (tmp.at<uchar>(y, x) == 255)
					{
						std::vector<cv::Point2i> line;
						myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
						if (!line.empty())
						{
							lines[count] = line;
							count++;
						}
					}
				}
			}
		}
		else if (mode == TOP_RIGHT || mode == BOTTOM_RIGHT)
		{
			for (int y = 2; y < height - 2; y++)
			{
				for (int x = width - 3; x > 1; x--)
				{
					if (tmp.at<uchar>(y, x) == 255)
					{
						std::vector<cv::Point2i> line;
						myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
						if (!line.empty())
						{
							lines[count] = line;
							count++;
						}
					}
				}
			}
		}
	}
	else
	{
		for (int x = 2; x < width - 2; x++)
		{
			for (int y = 2; y < height - 2; y++)
			{
				if (tmp.at<uchar>(y, x) == 255)
				{
					std::vector<cv::Point2i> line;
					myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
					if (!line.empty())
					{
						lines[count] = line;
						count++;
					}
				}
			}
		}
	}
}

int removeShortEdges2(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon,
	RIGHT_COUNT_SIDE mode)
{
	int width = src.cols;
	int height = src.rows;

	int count = 0;
	if (!lines.empty())
	{
		lines.clear();
	}

	cv:Mat tmp = src.clone();

	int maxLen = 0;
	if (isHorizon)
	{
		if (mode == TOP_LEFT || mode == BOTTOM_LEFT)
		{
			for (int y = 2; y < height - 2; y++)
			{
				for (int x = 2; x < width - 2; x++)
				{
					if (tmp.at<uchar>(y, x) == 255)
					{
						std::vector<cv::Point2i> line;
						myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
						if (!line.empty())
						{
							lines[count] = line;
							count++;
							if (maxLen < line.size())
							{
								maxLen = line.size();
							}
						}
					}
				}
			}
		}
		else if (mode == TOP_RIGHT || mode == BOTTOM_RIGHT)
		{
			for (int y = 2; y < height - 2; y++)
			{
				for (int x = width - 3; x > 1; x--)
				{
					if (tmp.at<uchar>(y, x) == 255)
					{
						std::vector<cv::Point2i> line;
						myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
						if (!line.empty())
						{
							lines[count] = line;
							count++;
							if (maxLen < line.size())
							{
								maxLen = line.size();
							}
						}
					}
				}
			}
		}
	}
	else
	{
		for (int x = 2; x < width - 2; x++)
		{
			for (int y = 2; y < height - 2; y++)
			{
				if (tmp.at<uchar>(y, x) == 255)
				{
					std::vector<cv::Point2i> line;
					myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
					if (!line.empty())
					{
						lines[count] = line;
						count++;
						if (maxLen < line.size())
						{
							maxLen = line.size();
						}
					}
				}
			}
		}
	}
	return maxLen;
}

void post_removeShortEdges2(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon, RIGHT_COUNT_SIDE mode)
{
	int width = src.cols;
	int height = src.rows;

	if(isHorizon)
	{
		for(auto it = lines.begin(); it != lines.end(); it++)
		{
			if(it->second.size() < lenThres)
			{
				for (std::vector<cv::Point2i>::iterator it_ = it->second.begin(); it_ != it->second.end(); it_++)
				{
					cv::Point2i pt = *it_;
					src.at<uchar>(pt.y, pt.x) = 0;
				}
			}
		}
	}
	else
	{
		for (auto it = lines.begin(); it != lines.end(); it++)
		{
			if (it->second.size() < lenThres)
			{
				for (std::vector<cv::Point2i>::iterator it_ = it->second.begin(); it_ != it->second.end(); it_++)
				{
					cv::Point2i pt = *it_;
					src.at<uchar>(pt.y, pt.x) = 0;
				}
			}
		}
	}

	removeShortEdges2(src, lines, lenThres, isHorizon, mode);
}

void post_process(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, bool isHorizon, RIGHT_COUNT_SIDE mode)
{
	connectEdge(src, 5, isHorizon);
	//int maxLen = removeShortEdges2(src, lines, 100, isHorizon, mode);
	//post_removeShortEdges2(src, lines, maxLen / 2, isHorizon, mode);
	int maxLen = removeShortEdges2(src, lines, 100, isHorizon, mode);
	connectEdge_(src, 10, isHorizon);
	post_removeShortEdges2(src, lines, maxLen/2, isHorizon, mode);
}

/**
 * \brief detect chess corners based on line detection
 * \param src 
 * \param pts 
 * \param ptsReal 
 */
void detectPts(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal, double grid_size)
{
	cv::Mat lineV, lineV_inv;
	cv::Mat lineH, lineH_inv;
	detectLines_(src[0], src[1], lineV, lineV_inv, false);
	detectLines_(src[2], src[3], lineH, lineH_inv, true);

	bitwise_and(lineV, lineV_inv, lineV);
	bitwise_and(lineH, lineH_inv, lineH);

	std::map<int, std::vector<cv::Point2i> > lines_H, lines_V;
	post_process(lineH, lines_H, true);
	post_process(lineV, lines_V, false);

	std::map<cv::Point2i, cv::Point2d, myCmp_map>  pts_H;

	cv::Mat ptsImg;
	bitwise_and(lineH, lineV, ptsImg);
	//cv::Mat ptsImg_2;
	//bitwise_not(ptsImg, ptsImg_2);

	int height = ptsImg.rows;
	int width = ptsImg.cols;
	int half_winsize = 3;
	cv::Mat tmp = ptsImg.clone();
	for(int y = 0; y < height; y++)
	{
		for(int x = 0; x < width; x++)
		{
			if(tmp.at<uchar>(y, x) == 255)
			{
				int h_key = -1;
				int v_key = -1;
				for(auto it = lines_H.begin(); 
					it != lines_H.end(); it++)
				{
					auto vec_it = find(it->second.begin(), it->second.end(), cv::Point2i(x, y));
					if(vec_it != it->second.end())
					{
						h_key = it->first;
						break;
					}
				}
				for (auto it = lines_V.begin();
					it != lines_V.end(); it++)
				{
					auto vec_it = find(it->second.begin(), it->second.end(), cv::Point2i(x, y));
					if (vec_it != it->second.end())
					{
						v_key = it->first;
						break;
					}
				}

				vector<Point2i> originPts;
				int num = 0;
				for(int winy = -half_winsize; winy <= half_winsize; winy++)
				{
					for(int winx = -half_winsize; winx <= half_winsize; winx++)
					{
						if(!(y + winy >= 0 && y + winy < height && x + winx >= 0 && x + winx < width))
						{
							continue;
						}
						if (tmp.at<uchar>(y + winy, x + winx) == 255)
						{
							tmp.at<uchar>(y + winy, x + winx) = 0;
							originPts.push_back(Point2i(x + winx, y + winy));
							num++;
						}
					}
				}
				cv::Point2f cornerPt;
				if(num > 1)
				{
					double sumX = 0.0, sumY = 0.0;
					for(int i = 0; i < originPts.size(); i++)
					{
						sumX += originPts[i].x;
						sumY += originPts[i].y;
					}
					cornerPt.x = sumX / num;
					cornerPt.y = sumY / num;
				}
				else if(num == 1)
				{
					cornerPt.x = x;
					cornerPt.y = y;
				}
				pts_H[cv::Point2i(h_key, v_key)] = cornerPt;
			}
		}
	}

	if(!pts.empty())
	{
		pts.clear();
	}
	for(auto it = pts_H.begin(); it != pts_H.end(); it++)
	{
		pts.push_back(it->second);
		ptsReal.push_back(cv::Point3f(it->first.x * grid_size * 1.0, it->first.y * grid_size * 1.0, 0));
	}
}

/**
 * \brief detecting the corners when the whole view is not avaliable
 * \param src 
 * \param pts 
 * \param ptsReal 
 * \param grid_size 
 * \param hNum
 * \param vNum
 * \param mode :indicate the complete side of the view
 */
void detectPts(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal,
	double grid_size, int hNum, int vNum, RIGHT_COUNT_SIDE mode, cv::Mat mask)
{
	cv::Mat lineV, lineV_inv;
	cv::Mat lineH, lineH_inv;
	detectLines_(src[2], src[3], lineV, lineV_inv, false);
	detectLines_(src[0], src[1], lineH, lineH_inv, true);

	bitwise_and(lineV, lineV_inv, lineV);
	bitwise_and(lineH, lineH_inv, lineH);

	if(!mask.empty())
	{
		bitwise_and(lineV, mask, lineV);
		bitwise_and(lineH, mask, lineH);
	}
	std::map<int, std::vector<cv::Point2i> > lines_H, lines_V;
	post_process(lineH, lines_H, true, mode);
	post_process(lineV, lines_V, false, mode);

	cout << lines_H.size() << endl;
	cout << lines_V.size() << endl;

	std::map<cv::Point2i, cv::Point2f, myCmp_map>  pts_H;

	cv::Mat ptsImg;
	bitwise_and(lineH, lineV, ptsImg);
	cv::Mat ptsImg_;
	bitwise_or(lineH, lineV, ptsImg_);

	int height = ptsImg.rows;
	int width = ptsImg.cols;

	int max_h = -1;
	int max_v = -1;

	int half_winsize = 3;
	cv::Mat tmp = ptsImg.clone();
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (tmp.at<uchar>(y, x) == 255)
			{
				int h_key = -1;
				int v_key = -1;
				for (auto it = lines_H.begin();
					it != lines_H.end(); it++)
				{
					auto vec_it = find(it->second.begin(), it->second.end(), cv::Point2i(x, y));
					if (vec_it != it->second.end())
					{
						h_key = it->first;
						break;
					}
				}
				for (auto it = lines_V.begin();
					it != lines_V.end(); it++)
				{
					auto vec_it = find(it->second.begin(), it->second.end(), cv::Point2i(x, y));
					if (vec_it != it->second.end())
					{
						v_key = it->first;
						break;
					}
				}

				if(max_h < h_key)
				{
					max_h = h_key;
				}
				if(max_v < v_key)
				{
					max_v = v_key;
				}

				vector<Point2i> originPts;
				int num = 0;
				for (int winy = -half_winsize; winy <= half_winsize; winy++)
				{
					for (int winx = -half_winsize; winx <= half_winsize; winx++)
					{
						if (!(y + winy >= 0 && y + winy < height && x + winx >= 0 && x + winx < width))
						{
							continue;
						}
						if (tmp.at<uchar>(y + winy, x + winx) == 255)
						{
							tmp.at<uchar>(y + winy, x + winx) = 0;
							originPts.push_back(Point2i(x + winx, y + winy));
							num++;
						}
					}
				}
				cv::Point2f cornerPt;
				if (num > 1)
				{
					double sumX = 0.0, sumY = 0.0;
					for (int i = 0; i < originPts.size(); i++)
					{
						sumX += originPts[i].x;
						sumY += originPts[i].y;
					}
					cornerPt.x = sumX / num;
					cornerPt.y = sumY / num;
				}
				else if (num == 1)
				{
					cornerPt.x = x;
					cornerPt.y = y;
				}
				pts_H[cv::Point2i(h_key, v_key)] = cornerPt;
			}
		}
	}

	if (!pts.empty())
	{
		pts.clear();
	}
	switch (mode)
	{
	case TOP_LEFT:
	{
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f(it->first.x * grid_size, it->first.y * grid_size, 0));
		}
	}
		break;
	case TOP_RIGHT:
	{
		int diff_v = vNum - 1 - max_v;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f(it->first.x * grid_size, (diff_v + it->first.y) * grid_size, 0));
		}
	}
		break;
	case BOTTOM_LEFT:
	{
		int diff_h = hNum - 1 - max_h;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f((diff_h + it->first.x) * grid_size, it->first.y * grid_size, 0));
		}
	}
		break;
	case BOTTOM_RIGHT:
	{
		int diff_h = hNum - 1 - max_h;
		int diff_v = vNum - 1 - max_v;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f((diff_h + it->first.x) * grid_size, (diff_v + it->first.y) * grid_size, 0));
		}
	}
		break;
	}

	//cv::Mat src_1, src_2, dst_1;
	//threshold(src[0], src_1, 80, 255, THRESH_BINARY);
	//threshold(src[2], src_2, 80, 255, THRESH_BINARY);
	//bitwise_xor(src_1, src_2, dst_1);

	cornerSubPix(src[4], pts, cv::Size(5, 5), cv::Size(-1, -1),
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 700, 1e-8));

}

void detectPts2(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal,
	double grid_size, int hNum, int vNum, RIGHT_COUNT_SIDE mode, cv::Mat mask)
{
	cv::Mat lineV, lineV_inv;
	cv::Mat lineH, lineH_inv;
	detectLines_(src[0], src[1], lineV, lineV_inv, false);
	detectLines_(src[2], src[3], lineH, lineH_inv, true);

	bitwise_and(lineV, lineV_inv, lineV);
	bitwise_and(lineH, lineH_inv, lineH);

	if (!mask.empty())
	{
		bitwise_and(lineV, mask, lineV);
		bitwise_and(lineH, mask, lineH);
	}
	std::map<int, std::vector<cv::Point2i> > lines_H, lines_V;
	post_process(lineH, lines_H, true, mode);
	post_process(lineV, lines_V, false, mode);

	std::map<cv::Point2i, cv::Point2f, myCmp_map>  pts_H;

	cv::Mat ptsImg;
	bitwise_and(lineH, lineV, ptsImg);
	cv::Mat ptsImg_;
	bitwise_or(lineH, lineV, ptsImg_);

	int height = ptsImg.rows;
	int width = ptsImg.cols;

	int max_h = lines_H.size() - 1;
	int max_v = lines_V.size() - 1;

	for(int h_i = 0; h_i <= max_h; h_i++)
	{
		cv::Mat h_img = cv::Mat::zeros(height, width, CV_8UC1);
		for(auto it = lines_H[h_i].begin(); it != lines_H[h_i].end(); it++)
		{
			h_img.at<uchar>(it->y, it->x) = 255;
		}
		for(int v_i = 0; v_i <= max_v; v_i++)
		{
			cv::Point key_pt(h_i, v_i);

			cv::Mat tmpImg = h_img.clone();
			for(auto it_ = lines_V[v_i].begin(); it_ != lines_V[v_i].end(); it_++)
			{
				tmpImg.at<uchar>(it_->y, it_->x) = 255;
			}

			cv::Mat cornerDst;
			cornerHarris(tmpImg, cornerDst, 15, 3, 0.04);
			cv::Point maxPt;
			minMaxLoc(cornerDst, NULL, NULL, NULL, &maxPt);

			pts_H[cv::Point(h_i, v_i)] = maxPt;
		}
	}
	if (!pts.empty())
	{
		pts.clear();
	}
	switch (mode)
	{
	case TOP_LEFT:
	{
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f(it->first.x * grid_size, it->first.y * grid_size, 0));
		}
	}
	break;
	case TOP_RIGHT:
	{
		int diff_v = vNum - 1 - max_v;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f(it->first.x * grid_size, (diff_v + it->first.y) * grid_size, 0));
		}
	}
	break;
	case BOTTOM_LEFT:
	{
		int diff_h = hNum - 1 - max_h;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f((diff_h + it->first.x) * grid_size, it->first.y * grid_size, 0));
		}
	}
	break;
	case BOTTOM_RIGHT:
	{
		int diff_h = hNum - 1 - max_h;
		int diff_v = vNum - 1 - max_v;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f((diff_h + it->first.x) * grid_size, (diff_v + it->first.y) * grid_size, 0));
		}
	}
	break;
	}

	cornerSubPix(ptsImg_, pts, cv::Size(5, 5), cv::Size(-1, -1),
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 700, 1e-8));
}

void loadXML_imgPath(std::string xmlPath, cv::Size& imgSize, map<RIGHT_COUNT_SIDE, vector<vector<std::string> > >& path_)
{
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xmlPath.c_str());
	tinyxml2::XMLElement *root = doc.FirstChildElement("all");
	imgSize.width = atoi(root->FirstChildElement("img_width")->GetText());
	imgSize.height = atoi(root->FirstChildElement("img_height")->GetText());


	vector<vector<std::string> > imgPath_tl;
	tinyxml2::XMLElement *root_tl = root->FirstChildElement("images_tl");
	tinyxml2::XMLElement *node_tl = root_tl->FirstChildElement("pair");
	while (node_tl) {
		vector<string> filenames(5);

		tinyxml2::XMLElement *filename = node_tl->FirstChildElement("pattern");
		int count;
		for (count = 0; count < 5; ++count) {
			if (!filename) {
				break;
			}
			filenames[count] = std::string(filename->GetText());
			filename = filename->NextSiblingElement("pattern");
		}
		imgPath_tl.push_back(filenames);
		node_tl = node_tl->NextSiblingElement("pair");
	}
	if(!imgPath_tl.empty())
	{
		path_[TOP_LEFT] = imgPath_tl;
	}


	vector<vector<std::string> > imgPath_tr;
	tinyxml2::XMLElement *root_tr = root->FirstChildElement("images_tr");
	tinyxml2::XMLElement *node_tr = root_tr->FirstChildElement("pair");
	while (node_tr) {
		vector<string> filenames(5);

		tinyxml2::XMLElement *filename = node_tr->FirstChildElement("pattern");
		int count;
		for (count = 0; count < 5; ++count) {
			if (!filename) {
				break;
			}
			filenames[count] = std::string(filename->GetText());
			filename = filename->NextSiblingElement("pattern");
		}
		imgPath_tr.push_back(filenames);
		node_tr = node_tr->NextSiblingElement("pair");
	}
	if (!imgPath_tr.empty())
	{
		path_[TOP_RIGHT] = imgPath_tr;
	}

	vector<vector<std::string> > imgPath_bl;
	tinyxml2::XMLElement *root_bl = root->FirstChildElement("images_bl");
	tinyxml2::XMLElement *node_bl = root_bl->FirstChildElement("pair");
	while (node_bl) {
		vector<string> filenames(5);

		tinyxml2::XMLElement *filename = node_bl->FirstChildElement("pattern");
		int count;
		for (count = 0; count < 5; ++count) {
			if (!filename) {
				break;
			}
			filenames[count] = std::string(filename->GetText());
			filename = filename->NextSiblingElement("pattern");
		}
		imgPath_bl.push_back(filenames);
		node_bl = node_bl->NextSiblingElement("pair");
	}
	if (!imgPath_bl.empty())
	{
		path_[BOTTOM_LEFT] = imgPath_bl;
	}

	vector<vector<std::string> > imgPath_br;
	tinyxml2::XMLElement *root_br = root->FirstChildElement("images_br");
	tinyxml2::XMLElement *node_br = root_br->FirstChildElement("pair");
	while (node_br) {
		vector<string> filenames(5);

		tinyxml2::XMLElement *filename = node_br->FirstChildElement("pattern");
		int count;
		for (count = 0; count < 5; ++count) {
			if (!filename) {
				break;
			}
			filenames[count] = std::string(filename->GetText());
			filename = filename->NextSiblingElement("pattern");
		}
		imgPath_br.push_back(filenames);
		node_br = node_br->NextSiblingElement("pair");
	}
	if (!imgPath_br.empty())
	{
		path_[BOTTOM_RIGHT] = imgPath_br;
	}
}

bool ptsCalib_single2(std::string xmlFilePath, cv::Size& imgSize, douVecPt2f& pts, douVecPt3f& ptsReal, double gridSize,
	int hNum, int vNum, cv::Mat mask)
{
	map<RIGHT_COUNT_SIDE, vector<vector<std::string> > > imgPaths;
	loadXML_imgPath(xmlFilePath, imgSize, imgPaths);

	if(!pts.empty())
	{
		pts.clear();
	}
	if(!ptsReal.empty())
	{
		ptsReal.clear();
	}

	for(auto it = imgPaths.begin(); it != imgPaths.end(); it++)
	{
		if (it->first == TOP_LEFT || it->first == TOP_RIGHT || it->first == BOTTOM_LEFT || it->first == BOTTOM_RIGHT)
		{

			for (auto it_1 = it->second.begin(); it_1 != it->second.end(); it_1++)
			{
				vector<cv::Point2f> oneImgPts;
				vector<cv::Point3f> oneObjPts;

				if ((*it_1).size() != 5)
				{
					continue;
				}

				vector<cv::Mat> oneImgs;
				for (int i = 0; i < 5; i++)
				{
					cv::Mat img = imread((*it_1)[i]);
					cvtColor(img, img, COLOR_BGR2GRAY);
					oneImgs.push_back(img);
				}

				detectPts(oneImgs, oneImgPts, oneObjPts, gridSize, hNum, vNum, it->first, mask);
				pts.push_back(oneImgPts);
				ptsReal.push_back(oneObjPts);
			}
		}
	}

	return(!(pts.empty() || ptsReal.empty() || pts.size() != ptsReal.size()));
}

/**
 * \brief stereo camera calibration for stereo vision with pre-captured images
 * \param imgFilePath :the path of image pairs captured by left and right cameras, for stereo camera calibration
 * \param cameraParaPath :the path of files storing the camera parameters
 */
double stereoCamCalibration(std::string imgFilePath, std::string cameraParaPath)
{
	//detect the inner corner in each chess image
	cv::Size imgSize;
	std::vector<std::vector<Point2f> > cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	std::vector<std::vector<Point3f> > objPts3d;					 	//calculated coordination of corners in world coordinate system
	bool isSuc = ptsCalib(imgFilePath, imgSize, cornerPtsVec_left, cornerPtsVec_right, objPts3d, 6, 9);
	if(!isSuc)
	{
		cout << "points detection failed." << endl;
		return -1;
	}

	//single camera calibration
	int flag = 0;
	flag |= CALIB_FIX_PRINCIPAL_POINT;
	//flag |= CALIB_FIX_ASPECT_RATIO;
	flag |= CALIB_ZERO_TANGENT_DIST;
	flag |= CALIB_RATIONAL_MODEL;
	//flag |= CALIB_FIX_K3;
	//flag |= CALIB_FIX_K4;
	//flag |= CALIB_FIX_K5;
	//left camera calibration
	cv::Mat K_left = cv::Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	cv::Mat D_left = cv::Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_left;										//matrix T of each image
	std::vector<cv::Mat> R_left;										//matrix R of each image
	double rmsLeft = calibrateCamera(objPts3d, cornerPtsVec_left, imgSize,
		K_left, D_left, 
		R_left, T_left, flag);
	//right camera calibration
	cv::Mat K_right = cv::Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	cv::Mat D_right = cv::Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_right;									//matrix T of each image
	std::vector<cv::Mat> R_right;									//matrix R of each image
	double rmsRight = calibrateCamera(objPts3d, cornerPtsVec_right, imgSize,
		K_right, D_right, 
		R_right, T_right, flag);

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	cv::Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	cv::Mat matrixE;				//essential matrix E
	cv::Mat matrixF;				//fundamental matrix F
	int stereoFlag = 0;
	stereoFlag |= CALIB_FIX_INTRINSIC;
	stereoFlag |= CALIB_SAME_FOCAL_LENGTH;

	double rms = stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		K_left, D_left,
		K_right, D_right,
		imgSize, matrixR, matrixT, matrixE, matrixF, stereoFlag,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0));

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "Imgcv::Size" << imgSize;
	fn << "Left_CameraInnerPara" << K_left;
	fn << "Left_CameraDistPara" << D_left;
	fn << "Right_CameraInnerPara" << K_right;
	fn << "Right_CameraDistPara" << D_right;
	fn << "R2L_Rotation_cv::Matrix" << matrixR;
	fn << "R2L_Translate_cv::Matrix" << matrixT;
	fn << "Essentialcv::Mat" << matrixE;
	fn << "Fundamentalcv::Mat" << matrixF;
	fn.release();

	return rms;
}

double stereoCamCalibration_2(std::string imgFilePathL, std::string cameraParaPath)
{
	//detect the inner corner in each chess image
	cv::Size imgSize;
	std::vector<std::vector<Point2f> > cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	std::vector<std::vector<Point3f> > objPts3d;					 	//calculated coordination of corners in world coordinate system
	bool isSuc = ptsCalib(imgFilePathL, imgSize, cornerPtsVec_left, cornerPtsVec_right, objPts3d, 6, 9);
	if (!isSuc)
	{
		cout << "points detection failed." << endl;
		return -1;
	}

	//single camera calibration
	int flag = 0;
	flag |= CALIB_FIX_PRINCIPAL_POINT;
	flag |= CALIB_FIX_ASPECT_RATIO;
	flag |= CALIB_ZERO_TANGENT_DIST;
	flag |= CALIB_RATIONAL_MODEL;
	flag |= CALIB_FIX_K3;
	flag |= CALIB_FIX_K4;
	flag |= CALIB_FIX_K5;
	//left camera calibration
	cv::Mat K_left = cv::Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	cv::Mat D_left = cv::Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_left;										//matrix T of each image
	std::vector<cv::Mat> R_left;										//matrix R of each image
	double rmsLeft = calibrateCamera(objPts3d, cornerPtsVec_left, imgSize,
		K_left, D_left, 
		R_left, T_left, flag);
	//right camera calibration
	cv::Mat K_right = cv::Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	cv::Mat D_right = cv::Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_right;									//matrix T of each image
	std::vector<cv::Mat> R_right;									//matrix R of each image
	double rmsRight = calibrateCamera(objPts3d, cornerPtsVec_right, imgSize,
		K_right, D_right, 
		R_right, T_right, flag);//| CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	cv::Mat monoMapL1, monoMapL2, monoMapR1, monoMapR2;
	//Precompute maps for cv::remap()
	fisheye::initUndistortRectifyMap(K_left, D_left, noArray(), K_left,
		imgSize, CV_32F, monoMapL1, monoMapL2);

	fisheye::initUndistortRectifyMap(K_right, D_right, noArray(), K_right,
		imgSize, CV_32F, monoMapR1, monoMapR2);


	for (int i = 0; i < cornerPtsVec_left.size(); i++)
	{
		fisheye::undistortPoints(cornerPtsVec_left[i], cornerPtsVec_left[i],
			K_left, D_left, noArray(), K_left);

		fisheye::undistortPoints(cornerPtsVec_right[i], cornerPtsVec_right[i],
			K_right, D_right, noArray(), K_right);
	}

	cv::Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	cv::Mat zeroDistortion = cv::Mat::zeros(D_right.size(), D_right.type());

	cv::Mat matrixE;				//essential matrix E
	cv::Mat matrixF;				//fundamental matrix F
	int stereoFlag = 0;
	stereoFlag |= CALIB_FIX_INTRINSIC;
	stereoFlag |= CALIB_SAME_FOCAL_LENGTH;

	double rms = stereoCalibrate(objPts3d, 
		cornerPtsVec_left, cornerPtsVec_right,
		K_left, zeroDistortion, K_right, zeroDistortion,
		imgSize, matrixR, matrixT, matrixE, matrixF, stereoFlag,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 50, 1e-6));

	std::cout << "stereo_calibration_error" << rms << std::endl;
	  
	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "Imgcv::Size" << imgSize;
	fn << "Left_CameraInnerPara" << K_left;
	fn << "Left_CameraDistPara" << D_left;
	fn << "Right_CameraInnerPara" << K_right;
	fn << "Right_CameraDistPara" << D_right;
	fn << "R2L_Rotation_cv::Matrix" << matrixR;
	fn << "R2L_Translate_cv::Matrix" << matrixT;
	fn << "Essentialcv::Mat" << matrixE;
	fn << "Fundamentalcv::Mat" << matrixF;
	fn.release();

	return rms;
}

/**
 * \brief single fisheye camera calibration
 * \param imgFilePath 
 * \param cameraParaPath 
 * \return 
 */
double fisheyeCamCalibSingle(std::string imgFilePath, std::string cameraParaPath)
{
	cv::Size imgSize;
	//std::vector<std::vector<Point2f> > cornerPtsVec;		//store the detected inner corners of each image
	//std::vector<std::vector<Point3f> > objPts3d;			//calculated coordination of corners in world coordinate system
	//bool isSuc = ptsCalib_Single(imgFilePath, imgSize, cornerPtsVec, objPts3d, 6, 9);
	//
	cv::Mat mask_;
	//createMask_lines(mask_);
	std::vector<std::vector<Point2f> > cornerPtsVec;		//store the detected inner corners of each image
	std::vector<std::vector<Point3f> > objPts3d;			//calculated coordination of corners in world coordinate system
	double gridSize = 16.5;
	bool isSuc = ptsCalib_single2(imgFilePath, imgSize, cornerPtsVec, objPts3d, gridSize, 17, 31, mask_);

	if (!isSuc)
	{
		cout << "points detection failed." << endl;
		return -1;
	}

	//single camera calibration
	int flag = 0;
	flag |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	//flag |= fisheye::CALIB_CHECK_COND;
	//flag |= fisheye::CALIB_FIX_SKEW;
	//flag |= fisheye::CALIB_FIX_K1;
	//flag |= fisheye::CALIB_FIX_K2;
	//flag |= fisheye::CALIB_FIX_K3;
	//flag |= fisheye::CALIB_FIX_K4;
	//flag |= fisheye::CALIB_FIX_PRINCIPAL_POINT;
	//flag |= fisheye::CALIB_FIX_INTRINSIC;
	flag |= fisheye::CALIB_USE_INTRINSIC_GUESS;

	//left camera calibration
	cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);		//the inner parameters of camera
	K.at<double>(0, 0) = 832.7025533;
	K.at<double>(1, 1) = 832.7025533;
	K.at<double>(0, 2) = 1280.0;
	K.at<double>(1, 2) = 720.0;
	cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);		//the paramters of camera distortion
	std::vector<cv::Mat> T;										//matrix T of each image:translation
	std::vector<cv::Mat> R;										//matrix R of each image:rotation
	double rms = cv::fisheye::calibrate(objPts3d, cornerPtsVec, imgSize,
		K, D, R, T, flag,
		cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 7000, 1e-10));// | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO 
	cout << "rms" << rms << endl;
	cout << K << endl;
	cout << D << endl;

	//if (rms < 1)
	//{
		FileStorage fn(cameraParaPath, FileStorage::WRITE);
		fn << "ImgSize" << imgSize;
		fn << "CameraInnerPara" << K;
		fn << "CameraDistPara" << D;
		fn << "RMS" << rms;
		fn.release();

		//distortRectify_fisheye(K, D, imgSize, imgFilePath);
	//}
	//else
	//{
	//	cout << "calibration failed" << endl;
	//}
	waitKey(0);

	return rms;
}

void distortRectify_fisheye(cv::Mat K, cv::Mat D, cv::Size imgSize, std::string imgFilePath)
{
	//load all the images in the folder
	String filePath = imgFilePath + "\\*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);

	int x_expand_half = 2560 * 0 / 2;
	int y_expand_half = 1440 * 0 / 2;
	Mat K_new = K;
	K_new.at<double>(0, 2) = K.at<double>(0, 2) + x_expand_half;
	K_new.at<double>(1, 2) = K.at<double>(1, 2) + y_expand_half;

	for (int i = 0; i < fileNames.size(); i++)
	{
		cv::Mat imgOrigin = imread(fileNames[i]);
		copyMakeBorder(imgOrigin, imgOrigin, y_expand_half, y_expand_half, x_expand_half, x_expand_half, BORDER_CONSTANT, 0);

		cv::Mat imgUndistort;

		my_cv::fisheye_r_d::undistortImage(imgOrigin, imgUndistort, K_new, D, K_new, imgSize);
		imwrite(fileNames[i].substr(0, fileNames[i].length() - 4) + "_undistort.jpg", imgUndistort);
	}
}

void distortRectify_fisheye(cv::Mat K, cv::Mat D, cv::Size imgSize, std::string imgFilePath,
	std::vector<cv::Mat>& undistortImgs, bool isLeft)
{
	if(!undistortImgs.empty())
	{
		undistortImgs.clear();
	}

	if (isLeft)
	{
		String filePath = imgFilePath + "\\*L.jpg";
		std::vector<String> fileNames;
		glob(filePath, fileNames, false);

		for (int i = 0; i < fileNames.size(); i++)
		{
			cv::Mat imgOrigin = imread(fileNames[i]);
			cv::Mat imgUndistort;
			fisheye::undistortImage(imgOrigin, imgUndistort, K, D, K, imgSize);
			undistortImgs.push_back(imgUndistort);
		}
	}
	else
	{
		String filePath = imgFilePath + "\\*R.jpg";
		std::vector<String> fileNames;
		glob(filePath, fileNames, false);

		for (int i = 0; i < fileNames.size(); i++)
		{
			cv::Mat imgOrigin = imread(fileNames[i]);
			cv::Mat imgUndistort;
			fisheye::undistortImage(imgOrigin, imgUndistort, K, D, K, imgSize);
			undistortImgs.push_back(imgUndistort);
		}
	}
}

/***********************************************************
 *************  fisheye camera calibration   ***************
 ***********************************************************
*/
/**
 * \brief directly using cv::fisheye::stereoCalibration() function to calibrate the stereo system
 * \param imgFilePathL 
 * \param cameraParaPath 
 */
double stereoFisheyeCamCalib(std::string imgFilePath, std::string cameraParaPath)
{
	//detect the inner corner in each chess image
	cv::Size imgSize;
	std::vector<std::vector<Point2f> > cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	std::vector<std::vector<Point3f> > objPts3d;					 	//calculated coordination of corners in world coordinate system
	bool isSuc = ptsCalib(imgFilePath, imgSize, cornerPtsVec_left, cornerPtsVec_right, objPts3d, 6, 9);
	if (!isSuc)
	{
		cout << "points detection failed." << endl;
		return -1;
	}

	cv::Mat K1, D1, K2, D2;
	cv::Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	cv::Mat matrixE;				//essential matrix E
	cv::Mat matrixF;				//fundamental matrix F
	int stereoFlag = 0;
	stereoFlag |= cv::fisheye::CALIB_FIX_INTRINSIC;
	stereoFlag |= cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
	stereoFlag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	stereoFlag |= cv::fisheye::CALIB_CHECK_COND;
	stereoFlag |= cv::fisheye::CALIB_FIX_SKEW;
	//stereoFlag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;
	//stereoFlag |= cv::fisheye::CALIB_FIX_K1;
	//stereoFlag |= cv::fisheye::CALIB_FIX_K2;
	//stereoFlag |= cv::fisheye::CALIB_FIX_K3;
	//stereoFlag |= cv::fisheye::CALIB_FIX_K4;

	double rms = fisheye::stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		K1, D1, K2, D2,
		imgSize, matrixR, matrixT, stereoFlag,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-12));

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "Imgcv::Size" << imgSize;
	fn << "Left_CameraInnerPara" << K1;
	fn << "Left_CameraDistPara" << D1;
	fn << "Right_CameraInnerPara" << K2;
	fn << "Right_CameraDistPara" << D2;
	fn << "R2L_Rotation_cv::Matrix" << matrixR;
	fn << "R2L_Translate_cv::Matrix" << matrixT;
	fn << "Essentialcv::Mat" << matrixE;
	fn << "Fundamentalcv::Mat" << matrixF;
	fn.release();

	return rms;
}

/**
 * \brief calibrate each fisheye camera first,then use the calculated parameters of cameras for further stereo calibration
 * \param imgFilePath
 * \param cameraParaPath 
 */
double stereoFisheyeCamCalib_2(std::string imgFilePath, std::string cameraParaPath)
{
	//detect the inner corner in each chess image
	cv::Size imgSize;
	std::vector<std::vector<Point2f> > cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	std::vector<std::vector<Point3f> > objPts3d;					 	//calculated coordination of corners in world coordinate system
	bool isSuc = ptsCalib(imgFilePath, imgSize, cornerPtsVec_left, cornerPtsVec_right, objPts3d, 6, 9);
	if(!isSuc)
	{
		cout << "points detection failed." << endl;
		return -1;
	}

	queue<int> toRemove_;
	//toRemove_.push(0);
	//toRemove_.push(0);
	//toRemove_.push(0);
	//toRemove_.push(0);
	//toRemove_.push(0);
	//toRemove_.push(0);
	//toRemove_.push(0);
	//toRemove_.push(9);
	//toRemove_.push(9);
	//toRemove_.push(11);
	//toRemove_.push(13);
	//toRemove_.push(13);
	//toRemove_.push(2);
	//toRemove_.push(9);
	while (!toRemove_.empty())
	{
		int curInt = toRemove_.front();
		toRemove_.pop();
		cornerPtsVec_left.erase(cornerPtsVec_left.begin() + curInt);
		cornerPtsVec_right.erase(cornerPtsVec_right.begin() + curInt);
		objPts3d.erase(objPts3d.begin() + curInt);
	}

	//*******************************************
	//******* single camera calibration *********
	//*******************************************
	int flag = 0;
	flag |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	//flag |= fisheye::CALIB_CHECK_COND;
	flag |= fisheye::CALIB_FIX_SKEW;
	//flag |= fisheye::CALIB_FIX_K1;
	//flag |= fisheye::CALIB_FIX_K2;
	//flag |= fisheye::CALIB_FIX_K3;
	//flag |= fisheye::CALIB_FIX_K4;
	//flag |= fisheye::CALIB_FIX_PRINCIPAL_POINT;
	//flag |= fisheye::CALIB_FIX_INTRINSIC;
	//flag |= fisheye::CALIB_USE_INTRINSIC_GUESS;

	//left camera calibration
	cv::Mat K_left = cv::Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	cv::Mat D_left = cv::Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_left;										//matrix T of each image:translation
	std::vector<cv::Mat> R_left;										//matrix R of each image:rotation
	double rmsLeft = fisheye::calibrate(objPts3d, cornerPtsVec_left, imgSize,
		K_left, D_left, R_left, T_left, flag,
		cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));// | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO 
	//right camera calibration
	cv::Mat K_right = cv::Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	cv::Mat D_right = cv::Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_right;									//matrix T of each image
	std::vector<cv::Mat> R_right;									//matrix R of each image
	double rmsRight = fisheye::calibrate(objPts3d, cornerPtsVec_right, imgSize,
		K_right, D_right, R_right, T_right, flag,
		cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));//| CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	//********************************************
	//************ stereo calibration ************
	//********************************************
	cv::Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	cv::Mat matrixE;				//essential matrix E
	cv::Mat matrixF;				//fundamental matrix F
	int stereoFlag = 0;
	stereoFlag |= cv::fisheye::CALIB_FIX_INTRINSIC;
	stereoFlag |= cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
	stereoFlag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	stereoFlag |= cv::fisheye::CALIB_CHECK_COND;
	stereoFlag |= cv::fisheye::CALIB_FIX_SKEW;
	//stereoFlag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;
	//stereoFlag |= cv::fisheye::CALIB_FIX_K1;
	//stereoFlag |= cv::fisheye::CALIB_FIX_K2;
	//stereoFlag |= cv::fisheye::CALIB_FIX_K3;
	//stereoFlag |= cv::fisheye::CALIB_FIX_K4;

	//toRemove_.push(9);
	//toRemove_.push(9);
	//toRemove_.push(11);
	//toRemove_.push(13);
	//toRemove_.push(13);
	//toRemove_.push(2);
	//toRemove_.push(9);
	while (!toRemove_.empty())
	{
		int curInt = toRemove_.front();
		toRemove_.pop();
		cornerPtsVec_left.erase(cornerPtsVec_left.begin() + curInt);
		cornerPtsVec_right.erase(cornerPtsVec_right.begin() + curInt);
		objPts3d.erase(objPts3d.begin() + curInt);
	}

	double rms = fisheye::stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		K_left, D_left, K_right, D_right,
		imgSize, matrixR, matrixT, stereoFlag,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6));

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "Imgcv::Size" << imgSize;
	fn << "Left_CameraInnerPara" << K_left;
	fn << "Left_CameraDistPara" << D_left;
	fn << "Right_CameraInnerPara" << K_right;
	fn << "Right_CameraDistPara" << D_right;
	fn << "R2L_Rotation_cv::Matrix" << matrixR;
	fn << "R2L_Translate_cv::Matrix" << matrixT;
	fn << "Essentialcv::Mat" << matrixE;
	fn << "Fundamentalcv::Mat" << matrixF;
	fn.release();

	return rms;
}

/**
 * \brief calibrate each camera first and un-distort the input images/points for further stereo calibration
 * \param imgFilePath
 * \param cameraParaPath 
 */
double stereoFisheyeCamCalib_3(std::string imgFilePath, std::string cameraParaPath)
{
	//detect the inner corner in each chess image
	cv::Size imgSize;
	std::vector<std::vector<Point2f> > cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	std::vector<std::vector<Point3f> > objPts3d;					 	//calculated coordination of corners in world coordinate system
	bool isSuc = ptsCalib(imgFilePath, imgSize, cornerPtsVec_left, cornerPtsVec_right, objPts3d, 6, 9);
	if (!isSuc)
	{
		cout << "points detection failed." << endl;
		return -1;
	}

	//*******************************************
	//******* single camera calibration *********
	//*******************************************
	int flag = 0;
	flag |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flag |= fisheye::CALIB_CHECK_COND;
	flag |= fisheye::CALIB_FIX_SKEW;
	//flag |= fisheye::CALIB_FIX_K1;
	//flag |= fisheye::CALIB_FIX_K2;
	//flag |= fisheye::CALIB_FIX_K3;
	//flag |= fisheye::CALIB_FIX_K4;
	//flag |= fisheye::CALIB_FIX_PRINCIPAL_POINT;
	//flag |= fisheye::CALIB_FIX_INTRINSIC;
	//flag |= fisheye::CALIB_USE_INTRINSIC_GUESS;

	//left camera calibration
	cv::Mat K_left = cv::Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	cv::Mat D_left = cv::Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_left;										//matrix T of each image:translation
	std::vector<cv::Mat> R_left;										//matrix R of each image:rotation
	double rmsLeft = fisheye::calibrate(objPts3d, cornerPtsVec_left, imgSize,
		K_left, D_left, R_left, T_left, flag,
		cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));
	//right camera calibration
	cv::Mat K_right = cv::Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	cv::Mat D_right = cv::Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<cv::Mat> T_right;									//matrix T of each image
	std::vector<cv::Mat> R_right;									//matrix R of each image
	double rmsRight = fisheye::calibrate(objPts3d, cornerPtsVec_right, imgSize,
		K_right, D_right, R_right, T_right, flag,
		cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));//

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	//********************************************
	//************ image undistortion and stereo calibration ************
	//********************************************
	if (rmsLeft < 1 && rmsRight < 1)
	{
		std::vector<std::vector<Point2f> > cornerPtsVecL_undist, cornerPtsVecR_undist;		//store the detected inner corners of each image
		std::vector<std::vector<Point3f> > objPts3d_;					 	//calculated coordination of corners in world coordinate system

		std::vector<cv::Mat> undistortImgL, undistortImgR;
		distortRectify_fisheye(K_left, D_left, imgSize, imgFilePath, undistortImgL, true);
		distortRectify_fisheye(K_right, D_right, imgSize, imgFilePath, undistortImgR, false);
		ptsCalib(undistortImgL, undistortImgR, cornerPtsVecL_undist, cornerPtsVecR_undist, objPts3d_, 6, 9);

		//********************************************
		//************ stereo calibration ************
		//********************************************
		cv::Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
		cv::Mat K1, K2, D1, D2;
		cv::Mat E, F, Q;
		int stereoFlag = 0;
		//stereoFlag |= cv::CALIB_USE_INTRINSIC_GUESS;
		//stereoFlag |= cv::CALIB_FIX_S1_S2_S3_S4;
		//stereoFlag |= cv::CALIB_ZERO_TANGENT_DIST;
		//stereoFlag |= cv::CALIB_FIX_INTRINSIC;
		//stereoFlag |= cv::CALIB_FIX_INTRINSIC;
		//stereoFlag |= cv::CALIB_FIX_INTRINSIC;
		//stereoFlag |= cv::CALIB_FIX_INTRINSIC;
		//stereoFlag |= cv::CALIB_FIX_INTRINSIC;
		//stereoFlag |= cv::CALIB_FIX_INTRINSIC;
		//stereoFlag |= cv::CALIB_FIX_INTRINSIC;

		//stereoFlag |= cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
		//stereoFlag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
		//stereoFlag |= cv::fisheye::CALIB_CHECK_COND;
		//stereoFlag |= cv::fisheye::CALIB_FIX_SKEW;
		//stereoFlag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;
		//stereoFlag |= cv::fisheye::CALIB_FIX_K1;
		//stereoFlag |= cv::fisheye::CALIB_FIX_K2;
		//stereoFlag |= cv::fisheye::CALIB_FIX_K3;
		//stereoFlag |= cv::fisheye::CALIB_FIX_K4;

		double rms = stereoCalibrate(objPts3d_,
			cornerPtsVecL_undist, cornerPtsVecR_undist,
			K1, D1, K2, D2,
			imgSize, matrixR, matrixT, E, F, Q, stereoFlag,
			cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 200, 1e-16));

		std::cout << "stereo_calibration_error" << rms << std::endl;

		cv::Mat R1, R2, P1, P2, Q_;
		stereoRectify(K1, D1, K2, D2,
			imgSize, matrixR, matrixT, R1, R2, P1, P2, Q_,
			CALIB_ZERO_DISPARITY);

		cv::Mat lmapx, lmapy, rmapx, rmapy;
		//rewrite for fisheye
		initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_32F, lmapx, lmapy);
		initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_32F, rmapx, rmapy);

		FileStorage fn(cameraParaPath, FileStorage::WRITE);
		fn << "FisheyeUndistort_K1" << K_left;
		fn << "FisheyeUndistort_D1" << D_left;
		fn << "FisheyeUndistort_K2" << K_left;
		fn << "FisheyeUndistort_D2" << D_left;
		fn << "Imgcv::Size" << imgSize;
		fn << "StereoCalib_K1" << K1;
		fn << "StereoCalib_D1" << D1;
		fn << "StereoCalib_K2" << K2;
		fn << "StereoCalib_D2" << D2;
		fn << "StereoCalib_R2L" << matrixR;
		fn << "StereoCalib_R2L" << matrixT;
		fn << "Rectify_R1" << R1;
		fn << "Rectify_P1" << P1;
		fn << "Rectify_R2" << R2;
		fn << "Rectify_P2" << P2;
		fn << "Map_mapxL" << lmapx;
		fn << "Map_mapxL" << lmapy;
		fn << "Map_mapxL" << rmapx;
		fn << "Map_mapxL" << rmapy;
		fn.release();

		return rms;
	}
}


/**
 * \brief merge the input four images to one big image:\merged
 * \param tl 
 * \param tr 
 * \param bl 
 * \param br 
 * \param merged 
 */
void merge4(const cv::Mat& tl, const cv::Mat& tr, const cv::Mat& bl, const cv::Mat& br, cv::Mat& merged)
{
	int type = tl.type();
	cv::Size sz = tl.size();
	if(type != tr.type() || type != bl.type() || type != br.type()
		|| sz.width != tr.cols || sz.width != bl.cols || sz.width != br.cols
		|| sz.height != tr.rows || sz.height != bl.rows || sz.height != br.rows)
	{
		cout<< "rectify failed."<<endl;
		return;
	}

	merged.create(cv::Size(sz.width * 2, sz.height * 2), type);
	tl.copyTo(merged(cv::Rect(0, 0, sz.width, sz.height)));
	tr.copyTo(merged(cv::Rect(sz.width, 0, sz.width, sz.height)));
	bl.copyTo(merged(cv::Rect(0, sz.height, sz.width, sz.height)));
	br.copyTo(merged(cv::Rect(sz.width, sz.height, sz.width, sz.height)));
}

/**
 * \brief using pre-calibrated parameters to rectify the images captured by left and right cameras in a real-time manner
 * \param cameraParaPath :the .xml file path of the pre-calculated camera calibration parameters
 */
void stereoFisheyeUndistort(cv::Mat distLeft, cv::Mat distRight, 
	std::string cameraParaPath, cv::Mat& rectiLeft, cv::Mat& rectiRight)
{
	//camera stereo calibration parameters
	cv::Size imgSize;
	cv::Mat leftK, leftD, rightK, rightD;
	//cv::Mat K1, D1, K2, D2;
	//cv::Mat R1, P1, R2, P2;
	//cv::Mat matrixR, matrixT;

	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["FisheyeUndistort_K1"] >> leftK;
	fn["FisheyeUndistort_D1"] >> leftD;
	fn["FisheyeUndistort_K2"] >> rightK;
	fn["FisheyeUndistort_D2"] >> rightD;
	fn["Imgcv::Size"] >> imgSize;
	//fn["StereoCalib_K1"] >> K1;
	//fn["StereoCalib_D1"] >> D1;
	//fn["StereoCalib_K2"] >> K2;
	//fn["StereoCalib_D2"] >> D2;
	//fn["StereoCalib_R2L"] >> matrixR;
	//fn["StereoCalib_R2L"] >> matrixT;
	//fn["Rectify_R1"] >> R1;
	//fn["Rectify_P1"] >> P1;
	//fn["Rectify_R2"] >> R2;
	//fn["Rectify_P2"] >> P2;
	//fn["Map_mapxL"] >> lmapx;
	//fn["Map_mapxL"] >> lmapy;
	//fn["Map_mapxL"] >> rmapx;
	//fn["Map_mapxL"] >> rmapy;
	fn.release();

	if(distLeft.size() != imgSize)
	{
		resize(distLeft, distLeft, imgSize);
	}
	if(distRight.size() != imgSize)
	{
		resize(distRight, distRight, imgSize);
	}
	cv::Mat undistortLeft, undistortRight;
	fisheye::undistortImage(distLeft, undistortLeft, leftK, leftD, leftK, imgSize);
	fisheye::undistortImage(distRight, undistortRight, rightK, rightD, rightK, imgSize);

	cv::Mat K1 = (cv::Mat_<double>(3,3) << 
		212.0527, 0, 0, 
		0, 210.5292, 0, 
		322.6531, 176.3494, 1);
	cv::Mat D1 = (cv::Mat_<double>(4, 1) <<
		0.0041, -0.0013, 0, 0);
	cv::Mat K2 = (cv::Mat_<double>(3, 3) <<
		215.5334, 0, 0,
		0, 211.0490, 0,
		323.6923,182.7369, 1);
	cv::Mat D2 = (cv::Mat_<double>(4, 1) <<
		-0.0053, 0.0025, 0, 0);
	cv::Mat matrixR = (cv::Mat_<double>(3, 3) <<
		0.998273142127583, -0.00292436388398383, -0.0586701099589566,
		0.00149299633958467, 0.999700528454590, -0.0244258954706668,
		0.0587239701370062, 0.0242961211613707, 0.997978553791543);
	cv::Mat matrixT = (cv::Mat_<double>(3, 1) <<
		-37.0476034022594, -7.32635604514172, 0.423312785199701);

	K1 = K1.t();
	K2 = K2.t();
	//matrixR = matrixR.t();

	cv::Mat R1, R2, P1, P2, Q_;
	stereoRectify(K1, D1, K2, D2,
		imgSize, matrixR, matrixT, R1, R2, P1, P2, Q_,
		CALIB_ZERO_DISPARITY);

	cv::Mat lmapx, lmapy, rmapx, rmapy;
	//rewrite for fisheye
	initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_32F, lmapx, lmapy);
	initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_32F, rmapx, rmapy);

	cv::remap(undistortLeft, rectiLeft, lmapx, lmapy, cv::INTER_LINEAR);
	cv::remap(undistortRight, rectiRight, rmapx, rmapy, cv::INTER_LINEAR);

	for (int ii = 0; ii < rectiLeft.rows; ii += 20)
	{
		cv::line(undistortLeft, cv::Point(0, ii), cv::Point(rectiLeft.cols, ii), cv::Scalar(0, 255, 0));
		cv::line(undistortRight, cv::Point(0, ii), cv::Point(rectiLeft.cols, ii), cv::Scalar(0, 255, 0));

		cv::line(rectiLeft, cv::Point(0, ii), cv::Point(rectiLeft.cols, ii), cv::Scalar(0, 255, 0));
		cv::line(rectiRight, cv::Point(0, ii), cv::Point(rectiLeft.cols, ii), cv::Scalar(0, 255, 0));
	}


	cv::Mat rectification;
	merge4(distLeft, distRight, rectiLeft, rectiRight, rectification);

	cv::imwrite("rectify.jpg", rectification);

}

void rectify_(std::string cameraParaPath, std::string imgPath)
{
	//camera stereo calibration parameters
	cv::Size imgSize;
	cv::Mat leftK, leftD, rightK, rightD;
	cv::Mat K1, D1, K2, D2;
	cv::Mat R1, P1, R2, P2;
	cv::Mat matrixR, matrixT;

	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["FisheyeUndistort_K1"] >> leftK;
	fn["FisheyeUndistort_D1"] >> leftD;
	fn["FisheyeUndistort_K2"] >> rightK;
	fn["FisheyeUndistort_D2"] >> rightD;
	fn["Imgcv::Size"] >> imgSize;
	fn["StereoCalib_K1"] >> K1;
	fn["StereoCalib_D1"] >> D1;
	fn["StereoCalib_K2"] >> K2;
	fn["StereoCalib_D2"] >> D2;
	fn["Rectify_R1"] >> R1;
	fn["Rectify_P1"] >> P1;
	fn["Rectify_R2"] >> R2;
	fn["Rectify_P2"] >> P2;
	fn.release();

	cv::Mat lmapx, lmapy, rmapx, rmapy;
	//rewrite for fisheye
	initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_32F, lmapx, lmapy);
	initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_32F, rmapx, rmapy);

	//load all the images in the folder
	String filePath = imgPath + "\\*L.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);

	for(int i = 0; i < fileNames.size(); i++)
	{
		cv::Mat distortImgL = imread(fileNames[i]);
		cv::Mat distortImgR = imread(fileNames[i].substr(0, fileNames[i].length() - 5) + "R.jpg");

		if(distortImgL.size() != imgSize)
		{
			resize(distortImgL, distortImgL, imgSize);
		}
		if (distortImgR.size() != imgSize)
		{
			resize(distortImgR, distortImgR, imgSize);
		}

		cv::Mat undistortLeft, undistortRight;
		fisheye::undistortImage(distortImgL, undistortLeft, leftK, leftD, leftK, imgSize);
		fisheye::undistortImage(distortImgR, undistortRight, rightK, rightD, rightK, imgSize);

		cv::Mat rectiLeft, rectiRight;
		cv::remap(undistortLeft, rectiLeft, lmapx, lmapy, cv::INTER_LINEAR);
		cv::remap(undistortRight, rectiRight, rmapx, rmapy, cv::INTER_LINEAR);

		for (int ii = 0; ii < rectiLeft.rows; ii += 100)
		{
			cv::line(undistortLeft, cv::Point(0, ii), cv::Point(rectiLeft.cols, ii), cv::Scalar(0, 255, 0));
			cv::line(undistortRight, cv::Point(0, ii), cv::Point(rectiLeft.cols, ii), cv::Scalar(0, 255, 0));

			cv::line(rectiLeft, cv::Point(0, ii), cv::Point(rectiLeft.cols, ii), cv::Scalar(0, 255, 0));
			cv::line(rectiRight, cv::Point(0, ii), cv::Point(rectiLeft.cols, ii), cv::Scalar(0, 255, 0));
		}


		cv::Mat rectification;
		merge4(undistortLeft, undistortRight, rectiLeft, rectiRight, rectification);

		cv::imwrite(fileNames[i].substr(0, fileNames[i].length() - 5) + "_rectify.jpg", rectification);
	}
}


/***********************************************************
 *****************  image rectification   ******************
 ***********************************************************
*/
cv::Mat mergeRectification(const cv::Mat& l, const cv::Mat& r)
{
	CV_Assert(l.type() == r.type() && l.size() == r.size());
	cv::Mat merged(l.rows, l.cols * 2, l.type());
	cv::Mat lpart = merged.colRange(0, l.cols);
	cv::Mat rpart = merged.colRange(l.cols, merged.cols);
	l.copyTo(lpart);
	r.copyTo(rpart);

	for (int i = 0; i < l.rows; i += 20)
		cv::line(merged, cv::Point(0, i), cv::Point(merged.cols, i), cv::Scalar(0, 255, 0));

	return merged;
}

/**
 * \brief using pre-calibrated parameters to rectify the images pre-captured by left and right cameras
 * \param imgFilePath
 * \param cameraParaPath
 */
void stereoCameraUndistort(std::string imgFilePath, std::string cameraParaPath)
{
	cv::Size imgSize;
	cv::Mat cameraInnerPara_left, cameraInnerPara_right;
	cv::Mat cameraDistPara_left, cameraDistPara_right;
	cv::Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["Imgcv::Size"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_cv::Matrix"] >> matrixR;
	fn["R2L_Translate_cv::Matrix"] >> matrixT;
	fn.release();

	cv::Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right;
	Rect validRoi[2];

	cv::Mat R_L, R_R, P_L, P_R, Q;
	fisheye::stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, R_L, R_R, P_L, P_R, Q,
		CALIB_ZERO_DISPARITY, imgSize, 0.0, 1.1);

	//// OpenCV can handle left-right
	//// or up-down camera arrangements
	//bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	cv::Mat lmapx, lmapy, rmapx, rmapy;
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	fisheye::initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left,
		R_L, P_L,
		imgSize * 2, CV_32F, lmapx, lmapy);

	fisheye::initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right,
		R_R, P_R,
		imgSize * 2, CV_32F, rmapx, rmapy);

	//cv::Mat canvas;
	//double sf;
	//int w, h;
	//if (!isVerticalStereo)
	//{
	//	sf = 600. / MAX(imgSize.width, imgSize.height);
	//	w = cvRound(imgSize.width*sf);
	//	h = cvRound(imgSize.height*sf);
	//	canvas.create(h, w * 2, CV_8UC3);
	//}
	//else
	//{
	//	sf = 300. / MAX(imgSize.width, imgSize.height);
	//	w = cvRound(imgSize.width*sf);
	//	h = cvRound(imgSize.height*sf);
	//	canvas.create(h * 2, w, CV_8UC3);
	//}

	//destroyAllWindows();


	//Stereo matching
	int ndisparities = 16 * 15;   /**< Range of disparity */
	int SADWindowSize = 31; /**< cv::Size of the block window. Must be odd */
	//Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	//sbm->setMinDisparity(0);			//确定匹配搜索从哪里开始，默认为0
	////sbm->setNumDisparities(64);		//在该数值确定的视差范围内进行搜索，视差窗口
	//								//即，最大视差值与最小视差值之差，大小必须为16的整数倍
	//sbm->setTextureThreshold(10);		//保证有足够的纹理以克服噪声
	//sbm->setDisp12MaxDiff(-1);			//左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异，默认为-1
	//sbm->setPreFilterCap(31);			//
	//sbm->setUniquenessRatio(25);		//使用匹配功能模式
	//sbm->setSpeckleRange(32);			//视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
	//sbm->setSpeckleWindowSize(100);		//检查视差连通区域变化度的窗口大小，值为0时取消speckle检查


	//Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 64, 7,
	//	10 * 7 * 7,
	//	40 * 7 * 7,
	//	1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);

	cv::Mat frame_left, frame_right;
	cv::Mat imgLeft, imgRight;
	cv::Mat rimg, cimg;
	cv::Mat Mask;

	String filePath = imgFilePath + "\\*L.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	for (int i = 0; i < fileNames.size(); i++)
	{
		frame_left = imread(fileNames[i]);
		frame_right = imread(fileNames[i].substr(0, fileNames[i].length() - 5) + "R.jpg");

		if (frame_left.rows != frame_right.rows
			&& frame_left.cols != frame_right.cols
			&& frame_left.rows != imgSize.height
			&& frame_left.cols != imgSize.width)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (frame_left.empty() || frame_right.empty())
			continue;

		//cv::rectangle(frame_left, cv::Rect(255, 0, 829, frame_left.rows - 1), cv::Scalar(0, 0, 255));
		//cv::rectangle(frame_right, cv::Rect(255, 0, 829, frame_left.rows - 1), cv::Scalar(0, 0, 255));
		//cv::rectangle(frame_right, cv::Rect(255 - ndisparities, 0, 829 + ndisparities, frame_left.rows - 1), cv::Scalar(0, 0, 255));

		cv::Mat lundist, rundist;
		cv::remap(frame_left, lundist, lmapx, lmapy, INTER_LINEAR);
		cv::remap(frame_right, rundist, rmapx, rmapy, cv::INTER_LINEAR);

		cv::Mat rectification = mergeRectification(lundist, rundist);

		//imgLeft = canvasPart1(vroi).clone();
		//imgRight = canvasPart2(vroi).clone();

		//rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
		//rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

		//if (!isVerticalStereo)
		//	for (int j = 0; j < canvas.rows; j += 32)
		//		line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		//else
		//	for (int j = 0; j < canvas.cols; j += 32)
		//		line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

		//cv::Mat imgLeftBGR = imgLeft.clone();
		//cv::Mat imgRightBGR = imgRight.clone();

		//cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
		//cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);


		char c = (char)waitKey(0);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

void getRectifiedImages(std::string imgFilePath, std::string cameraParaPath)
{
	cv::Size imgSize;
	cv::Mat cameraInnerPara_left, cameraInnerPara_right;
	cv::Mat cameraDistPara_left, cameraDistPara_right;
	cv::Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["Imgcv::Size"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_cv::Matrix"] >> matrixR;
	fn["R2L_Translate_cv::Matrix"] >> matrixT;
	fn.release();


	cv::Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q;
	Rect validRoi[2];

	stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q,
		CALIB_ZERO_DISPARITY, 0, imgSize, &validRoi[0], &validRoi[1]);

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	cv::Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left, matrixRectify_left, matrixProjection_left, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right, matrixRectify_right, matrixProjection_right, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	cv::Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3);
	}

	destroyAllWindows();


	//Stereo matching
	int ndisparities = 16 * 5;   /**< Range of disparity */
	int SADWindowSize = 31; /**< cv::Size of the block window. Must be odd */
	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	sbm->setMinDisparity(0);			//确定匹配搜索从哪里开始，默认为0
										//sbm->setNumDisparities(64);		//在该数值确定的视差范围内进行搜索，视差窗口
										//即，最大视差值与最小视差值之差，大小必须为16的整数倍
	sbm->setTextureThreshold(10);		//保证有足够的纹理以克服噪声
	sbm->setDisp12MaxDiff(-1);			//左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异，默认为-1
	sbm->setPreFilterCap(31);			//
	sbm->setUniquenessRatio(25);		//使用匹配功能模式
	sbm->setSpeckleRange(32);			//视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
	sbm->setSpeckleWindowSize(100);		//检查视差连通区域变化度的窗口大小，值为0时取消speckle检查


	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 64, 7,
		10 * 7 * 7,
		40 * 7 * 7,
		1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);

	cv::Mat frame_left, frame_right;
	cv::Mat imgLeft, imgRight;
	cv::Mat rimg, cimg;
	cv::Mat Mask;

	String filePath = imgFilePath + "/left*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	for (int i = 0; i < fileNames.size(); i++)
	{
		frame_left = imread(fileNames[i]);
		frame_right = imread(fileNames[i].substr(0, fileNames[i].length() - 10) + "right"
			+ fileNames[i].substr(fileNames[i].length() - 6, 6));

		if (frame_left.rows != frame_right.rows && frame_left.cols != frame_right.cols && frame_left.rows != imgSize.height && frame_left.cols != imgSize.width)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (frame_left.empty() || frame_right.empty())
			continue;

		remap(frame_left, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		cv::Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
		resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
		Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
			cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

		remap(frame_right, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		cv::Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));
		resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
		Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
			cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

		Rect vroi = vroi1 & vroi2;

		imgLeft = canvasPart1(vroi).clone();
		imgRight = canvasPart2(vroi).clone();

		rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
		rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

		if (!isVerticalStereo)
			for (int j = 0; j < canvas.rows; j += 32)
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (int j = 0; j < canvas.cols; j += 32)
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

		cv::Mat imgLeftBGR = imgLeft.clone();
		cv::Mat imgRightBGR = imgRight.clone();

		cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
		cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);


		//-- And create the image in which we will save our disparities
		cv::Mat imgDisparity16S = cv::Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		cv::Mat imgDisparity8U = cv::Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		cv::Mat sgbmDisp16S = cv::Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		cv::Mat sgbmDisp8U = cv::Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl;
			return;
		}

		sbm->compute(imgLeft, imgRight, imgDisparity16S);

		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
		applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
		cv::Mat disparityShow;
		imgDisparity8U.copyTo(disparityShow, Mask);


		sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

		sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
		applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
		cv::Mat  sgbmDisparityShow;
		sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

		imshow("bmDisparity", disparityShow);
		imshow("sgbmDisparity", sgbmDisparityShow);
		imshow("rectified", canvas);

		char c = (char)waitKey(0);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

void getRectifiedImages(cv::Mat imgLeft, cv::Mat imgRight, std::string cameraParaPath, cv::Mat& rectifiedLeft,
	cv::Mat& rectifiedRight)
{
	cv::Size imgSize;
	cv::Mat cameraInnerPara_left, cameraInnerPara_right;
	cv::Mat cameraDistPara_left, cameraDistPara_right;
	cv::Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["Imgcv::Size"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_cv::Matrix"] >> matrixR;
	fn["R2L_Translate_cv::Matrix"] >> matrixT;
	fn.release();


	cv::Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q;
	Rect validRoi[2];

	stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q,
		CALIB_ZERO_DISPARITY, 0, imgSize, &validRoi[0], &validRoi[1]);

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	cv::Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left, matrixRectify_left, matrixProjection_left, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right, matrixRectify_right, matrixProjection_right, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	cv::Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3);
	}

	cv::Mat rimg, cimg;
	cv::Mat Mask;

	if (imgLeft.rows != imgRight.rows && imgLeft.cols != imgRight.cols && imgLeft.rows != imgSize.height && imgLeft.cols != imgSize.width)
	{
		std::cout << "img reading error" << std::endl;
		return;
	}

	if (imgLeft.empty() || imgRight.empty())
		return;

	remap(imgLeft, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
	rimg.copyTo(cimg);
	cv::Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
	resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
	Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
		cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

	remap(imgRight, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
	rimg.copyTo(cimg);
	cv::Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));
	resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
	Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
		cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

	Rect vroi = vroi1 & vroi2;

	rectifiedLeft = canvasPart1(vroi).clone();
	rectifiedRight = canvasPart2(vroi).clone();

	rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
	rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

	if (!isVerticalStereo)
		for (int j = 0; j < canvas.rows; j += 32)
			line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
	else
		for (int j = 0; j < canvas.cols; j += 32)
			line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
}
