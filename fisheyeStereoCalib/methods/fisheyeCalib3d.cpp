#include "fisheyeCalib3d.h"
#include "parametersStereo.h"

using namespace cv;
using namespace std;

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
	Size imgSize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));

	Size patternSize(5, 7);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;


	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec;		//store the detected inner corners of each image
	Mat img;
	Mat imgGrey;
	while (cornerPtsVec.size() < 20)
	{
		cap.read(img);
		cvtColor(img, imgGrey, COLOR_BGR2GRAY);

		std::vector<Point2f> cornerPts;
		bool patternFound = findChessboardCorners(imgGrey, patternSize, cornerPts, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
			+ CALIB_CB_FAST_CHECK);
		if (patternFound)
		{
			cornerSubPix(imgGrey, cornerPts, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
			cornerPtsVec.push_back(cornerPts);
			drawChessboardCorners(imgGrey, patternSize, cornerPts, patternFound);
			cornerPtsVec.push_back(img);
		}
	}
	cap.release();


	//camera calibration
	Size2f squareSize(35.0, 36.2);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	Mat cameraMatrixInnerPara = Mat(3, 3, CV_32FC1, Scalar::all(0));	//the inner parameters of camera
	Mat cameraMatrixDistPara = Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT;									//matrix T of each image
	std::vector<Mat> vectorMatR;									//matrix R of each image

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

	calibrateCamera(objPts3d, cornerPtsVec, imgSize, cameraMatrixInnerPara, cameraMatrixDistPara, vectorMatR, vectorMatT, 0);


	//evaluate the result of the camera calibration,calculate the error of calibration in each image
	double totalErr = 0.0;
	double err = 0.0;
	std::vector<Point2f> imgPts_2d;		//store the rechecked points' coordination
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		std::vector<Point3f> tempPts = objPts3d[i];  //the actual coordination of point in 3d corrdinate system
		projectPoints(tempPts, vectorMatR[i], vectorMatT[i], cameraMatrixInnerPara, cameraMatrixDistPara, imgPts_2d);

		//calculate the error
		std::vector<Point2f> tempImagePoint = cornerPtsVec[i]; //the detected corner coordination in the image
		Mat tempImgPt = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat recheckImgPt = Mat(1, imgPts_2d.size(), CV_32FC2);

		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			recheckImgPt.at<Vec2f>(0, j) = Vec2f(imgPts_2d[j].x, imgPts_2d[j].y);
			tempImgPt.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(recheckImgPt, tempImgPt, NORM_L2);
		totalErr += err / gridPatternNum;
	}

	std::cout << "总平均误差：" << totalErr / cornerPtsVec.size() << std::endl;


	//output the calibration result
	std::cout << "相机内参数矩阵：\n" << cameraMatrixInnerPara << std::endl;
	std::cout << "相机畸变参数[k1,k2,k3,p1,p2]:\n" << cameraMatrixDistPara << std::endl;
	Mat rotationMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		Rodrigues(vectorMatR[i], rotationMatrix);
		std::cout << "第" << i + 1 << "幅图像的旋转矩阵：\n" << rotationMatrix << std::endl << std::endl;
		std::cout << "第" << i + 1 << "幅图像的平移矩阵：\n" << vectorMatT[i] << std::endl << std::endl << std::endl;
	}

	//store the calibration result to the .xml file
	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "CameraInnerPara" << cameraMatrixInnerPara;
	fn << "CameraDistPara" << cameraMatrixDistPara;
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
	Size patternSize(9, 6);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = 54;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec;		//store the detected inner corners of each image
	for (int i = 0; i < fileNames.size(); i++)
	{
		Mat img = imread(fileNames[i], IMREAD_GRAYSCALE);
		if (i == 0)
		{
			imgSize.width = img.rows;
			imgSize.height = img.cols;
		}

		std::vector<Point2f> cornerPts;
		bool patternFound = findChessboardCorners(img, patternSize, cornerPts, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
			+ CALIB_CB_FAST_CHECK);
		if (patternFound)
		{
			cornerSubPix(img, cornerPts, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
			cornerPtsVec.push_back(cornerPts);
			drawChessboardCorners(img, patternSize, cornerPts, patternFound);
		}
	}


	//camera calibration
	Size2f squareSize(35.0, 36.2);		//the real size of each grid in the chess board,which is measured manually by ruler
	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	Mat cameraMatrixInnerPara = Mat(3, 3, CV_32FC1, Scalar::all(0));	//the inner parameters of camera
	Mat cameraMatrixDistPara = Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT;									//matrix T of each image
	std::vector<Mat> vectorMatR;									//matrix R of each image

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

	calibrateCamera(objPts3d, cornerPtsVec, imgSize, cameraMatrixInnerPara, cameraMatrixDistPara, vectorMatR, vectorMatT, 0);


	//evaluate the result of the camera calibration,calculate the error of calibration in each image
	double totalErr = 0.0;
	double err = 0.0;
	std::vector<Point2f> imgPts_2d;		//store the rechecked points' coordination
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		std::vector<Point3f> tempPts = objPts3d[i];  //the actual coordination of point in 3d corrdinate system
		projectPoints(tempPts, vectorMatR[i], vectorMatT[i], cameraMatrixInnerPara, cameraMatrixDistPara, imgPts_2d);

		//calculate the error
		std::vector<Point2f> tempImagePoint = cornerPtsVec[i]; //the detected corner coordination in the image
		Mat tempImgPt = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat recheckImgPt = Mat(1, imgPts_2d.size(), CV_32FC2);

		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			recheckImgPt.at<Vec2f>(0, j) = Vec2f(imgPts_2d[j].x, imgPts_2d[j].y);
			tempImgPt.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(recheckImgPt, tempImgPt, NORM_L2);
		totalErr += err / gridPatternNum;
	}

	std::cout << "总平均误差：" << totalErr / cornerPtsVec.size() << std::endl;


	//output the calibration result
	std::cout << "相机内参数矩阵：\n" << cameraMatrixInnerPara << std::endl;
	std::cout << "相机畸变参数[k1,k2,k3,p1,p2]:\n" << cameraMatrixDistPara << std::endl;
	Mat rotationMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	for (int i = 0; i < cornerPtsVec.size(); i++)
	{
		Rodrigues(vectorMatR[i], rotationMatrix);
		std::cout << "第" << i + 1 << "幅图像的旋转矩阵：\n" << rotationMatrix << std::endl << std::endl;
		std::cout << "第" << i + 1 << "幅图像的平移矩阵：\n" << vectorMatT[i] << std::endl << std::endl << std::endl;
	}

	//store the calibration result to the .xml file
	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "CameraInnerPara" << cameraMatrixInnerPara;
	fn << "CameraDistPara" << cameraMatrixDistPara;
	fn.release();
}

/**
 * \brief using pre-calibrated camera inner parameters and distort parameters to undistort the images captured by the camera
 * \param cameraParaPath :the .xml file path of pre-calculated parameters of camera
 */
void myCameraUndistort(std::string cameraParaPath)
{
	Mat cameraMatrixInnerPara;
	Mat cameraMatrixDistPara;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["CameraInnerPara"] >> cameraMatrixInnerPara;
	fn["CameraDistPara"] >> cameraMatrixDistPara;
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
		Mat img;
		Mat undistortImg;
		cap.read(img);
		imshow("originView", img);
		undistort(img, undistortImg, cameraMatrixInnerPara, cameraMatrixDistPara);
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
	Mat cameraMatrixInnerPara;
	Mat cameraMatrixDistPara;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["CameraInnerPara"] >> cameraMatrixInnerPara;
	fn["CameraDistPara"] >> cameraMatrixDistPara;
	fn.release();

	String filePath = imgFilePath + "/*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	Mat undistortImg;
	std::vector<Mat> undistortImgs;
	for (int i = 0; i < fileNames.size(); i++)
	{
		Mat img = imread(fileNames[i]);
		undistort(img, undistortImg, cameraMatrixInnerPara, cameraMatrixDistPara);
		imwrite(imgFilePath + "/" + std::to_string(i) + ".jpg", undistortImg);
		undistortImgs.push_back(undistortImg);
	}
}

/**
 * \brief two camera calibration for stereo vision by openning camera and capturing the chess board images
 * \param cameraParaPath
 */
void twoCamerasCalibration(std::string cameraParaPath)
{
	//openning the two cameras:left camera, right camera
	VideoCapture cap_left, cap_right;
	cap_left.open(0);
	if (!cap_left.isOpened())
	{
		std::cout << "Left camera open failed!" << std::endl;
		return;
	}
	cap_left.set(CAP_PROP_FOURCC, 'GPJM');
	cap_left.set(CAP_PROP_FRAME_HEIGHT, 480);		//rows
	cap_left.set(CAP_PROP_FRAME_WIDTH, 640);			//cols

	cap_right.open(1);
	if (!cap_right.isOpened())
	{
		std::cout << "Left camera open failed!" << std::endl;
		return;
	}
	cap_right.set(CAP_PROP_FOURCC, 'GPJM');
	cap_right.set(CAP_PROP_FRAME_HEIGHT, cap_left.get(CAP_PROP_FRAME_HEIGHT));
	cap_right.set(CAP_PROP_FRAME_WIDTH, cap_left.get(CAP_PROP_FRAME_WIDTH));

	Size imgSize(cap_left.get(CAP_PROP_FRAME_WIDTH), cap_left.get(CAP_PROP_FRAME_HEIGHT));


	Size patternSize(9, 6);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;


	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	Mat img_left, img_right;
	Mat imgGrey_left, imgGrey_right;
	while (cornerPtsVec_left.size() < 20)
	{
		cap_left.read(img_left);
		cvtColor(img_left, imgGrey_left, COLOR_BGR2GRAY);
		cap_right.read(img_right);
		cvtColor(img_right, imgGrey_right, COLOR_BGR2GRAY);

		//Mat resizeLeft;
		//Mat resizeRight;
		//resize(imgGrey_left, resizeLeft, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);
		//resize(imgGrey_right, resizeRight, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);

		std::vector<Point2f> cornerPts_left;
		std::vector<Point2f> cornerPts_right;

		bool patternFound_left = findChessboardCorners(imgGrey_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			+ CALIB_CB_NORMALIZE_IMAGE);
		bool patternFound_right = findChessboardCorners(imgGrey_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			+ CALIB_CB_NORMALIZE_IMAGE);

		if (patternFound_left && patternFound_right)
		{
			//for (int k = 0; k < cornerPts_left.size(); k++)
			//{
			//	cornerPts_left[k].x /= 2.0;
			//	cornerPts_left[k].y /= 2.0;
			//}

			//for (int k = 0; k < cornerPts_right.size(); k++)
			//{
			//	cornerPts_right[k].x /= 2.0;
			//	cornerPts_right[k].y /= 2.0;
			//}

			cornerSubPix(imgGrey_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_left.push_back(cornerPts_left);
			drawChessboardCorners(imgGrey_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(imgGrey_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_right.push_back(cornerPts_right);
			drawChessboardCorners(imgGrey_right, patternSize, cornerPts_right, patternFound_right);
		}
	}
	cap_left.release();
	cap_right.release();

	//two cameras calibration
	Size2f squareSize(35.0, 36.2);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	for (std::vector<std::vector<Point2f>>::iterator itor = cornerPtsVec_left.begin(); itor != cornerPtsVec_left.end(); itor++)
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

	Mat cameraMatrixInnerPara_left = Mat(3, 3, CV_32FC1, Scalar::all(0));		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image
	std::vector<Mat> vectorMatR_left;										//matrix R of each image
	double rmsLeft = calibrateCamera(objPts3d, cornerPtsVec_left, imgSize,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST
		| CALIB_RATIONAL_MODEL | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);

	Mat cameraMatrixInnerPara_right = Mat(3, 3, CV_32FC1, Scalar::all(0));	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 5, CV_32FC1, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = calibrateCamera(objPts3d, cornerPtsVec_right, imgSize,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right,
		CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST
		| CALIB_RATIONAL_MODEL | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);

	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat matrixE;				//essential matrix E
	Mat matrixF;				//fundamental matrix F
	double rms = stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
		imgSize, matrixR, matrixT, matrixE, matrixF, CALIB_FIX_INTRINSIC | CALIB_SAME_FOCAL_LENGTH,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0));
	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << cameraMatrixInnerPara_left;
	fn << "Left_CameraDistPara" << cameraMatrixDistPara_left;
	fn << "Right_CameraInnerPara" << cameraMatrixInnerPara_right;
	fn << "Right_CameraDistPara" << cameraMatrixDistPara_right;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn << "Essential_Matrix" << matrixE;
	fn << "Fundamental_Matrix" << matrixF;
	fn.release();

	double err = 0;
	int npoints = 0;
	std::vector<Vec3f> lines[2];
	for (int i = 0; i < cornerPtsVec_left.size(); i++)
	{
		int npt = (int)cornerPtsVec_left[i].size();
		Mat imgpt[2];
		imgpt[0] = Mat(cornerPtsVec_left[i]);
		undistortPoints(imgpt[0], imgpt[0], cameraMatrixInnerPara_left, cameraMatrixDistPara_left, Mat(), cameraMatrixInnerPara_left);
		computeCorrespondEpilines(imgpt[0], 0 + 1, matrixF, lines[0]);

		imgpt[1] = Mat(cornerPtsVec_right[i]);
		undistortPoints(imgpt[1], imgpt[1], cameraMatrixInnerPara_right, cameraMatrixDistPara_right, Mat(), cameraMatrixInnerPara_right);
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
 * \brief two camera calibration for stereo vision with pre-captured images
 * \param imgFilePath :the path of image pairs captured by left and right cameras, for stereo camera calibration
 * \param cameraParaPath :the path of files storing the camera parameters
 */
void twoCamerasCalibration(std::string imgFilePath, std::string cameraParaPath)
{
	//read the two cameras' pre-captured  images:left camera, right camera
	//load all the images in the folder
	String filePath = imgFilePath + "/cal_left_*.jpg";
	std::vector<String> fileNames;
	glob(filePath, fileNames, false);
	Size patternSize(8, 6);		//5:the number of inner corners in each row of the chess board
								//7:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	for (int i = 0; i < fileNames.size(); i++)
	{
		Mat img_left = imread(fileNames[i], IMREAD_GRAYSCALE);
		Mat img_right = imread(fileNames[i].substr(0, fileNames[i].length() - 11) + "right_"
			+ fileNames[i].substr(fileNames[i].length() - 6, 6), IMREAD_GRAYSCALE);

		if (img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (i == 0)
		{
			imgSize.width = img_left.rows;
			imgSize.height = img_left.cols;
		}

		//Mat resizeLeft;
		//Mat resizeRight;
		//resize(img_left, resizeLeft, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);
		//resize(img_right, resizeRight, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(img_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE);
		bool patternFound_right = findChessboardCorners(img_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE);
		if (patternFound_left && patternFound_right)
		{
			//for(int k = 0; k < cornerPts_left.size(); k++)
			//{
			//	cornerPts_left[k].x /= 2.0;
			//	cornerPts_left[k].y /= 2.0;
			//}

			//for (int k = 0; k < cornerPts_right.size(); k++)
			//{
			//	cornerPts_right[k].x /= 2.0;
			//	cornerPts_right[k].y /= 2.0;
			//}

			cornerSubPix(img_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_left.push_back(cornerPts_left);
			drawChessboardCorners(img_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(img_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_right.push_back(cornerPts_right);
			drawChessboardCorners(img_right, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	Size2f squareSize(20.2222, 38.2857);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	for (std::vector<std::vector<Point2f>>::iterator itor = cornerPtsVec_left.begin(); itor != cornerPtsVec_left.end(); itor++)
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

	Mat cameraMatrixInnerPara_left = Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image
	std::vector<Mat> vectorMatR_left;										//matrix R of each image
	double rmsLeft = calibrateCamera(objPts3d, cornerPtsVec_left, imgSize,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST
		| CALIB_RATIONAL_MODEL | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);

	Mat cameraMatrixInnerPara_right = Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = calibrateCamera(objPts3d, cornerPtsVec_right, imgSize,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right,
		CALIB_FIX_PRINCIPAL_POINT | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST
		| CALIB_RATIONAL_MODEL | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat matrixE;				//essential matrix E
	Mat matrixF;				//fundamental matrix F
	double rms = stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
		imgSize, matrixR, matrixT, matrixE, matrixF, CALIB_FIX_INTRINSIC | CALIB_SAME_FOCAL_LENGTH,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0));

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << cameraMatrixInnerPara_left;
	fn << "Left_CameraDistPara" << cameraMatrixDistPara_left;
	fn << "Right_CameraInnerPara" << cameraMatrixInnerPara_right;
	fn << "Right_CameraDistPara" << cameraMatrixDistPara_right;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn << "EssentialMat" << matrixE;
	fn << "FundamentalMat" << matrixF;
	fn.release();

	double err = 0;
	int npoints = 0;
	std::vector<Vec3f> lines[2];
	for (int i = 0; i < cornerPtsVec_left.size(); i++)
	{
		int npt = (int)cornerPtsVec_left[i].size();
		Mat imgpt[2];
		imgpt[0] = Mat(cornerPtsVec_left[i]);
		undistortPoints(imgpt[0], imgpt[0], cameraMatrixInnerPara_left, cameraMatrixDistPara_left, Mat(), cameraMatrixInnerPara_left);
		computeCorrespondEpilines(imgpt[0], 0 + 1, matrixF, lines[0]);

		imgpt[1] = Mat(cornerPtsVec_right[i]);
		undistortPoints(imgpt[1], imgpt[1], cameraMatrixInnerPara_right, cameraMatrixDistPara_right, Mat(), cameraMatrixInnerPara_right);
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

void twoCamerasCalibration(std::string imgFilePathL, std::string imgFilePathR, std::string cameraParaPath)
{
	//read the two cameras' pre-captured  images:left camera, right camera
//load all the images in the folder
	String filePathL = imgFilePathL + "/*L.jpg";
	std::vector<String> fileNamesL;
	glob(filePathL, fileNamesL, false);
	Size patternSize(9, 6);		//表示棋盘行和列的内角数（行，列）	//9:the number of inner corners in each row of the chess board
																					//6:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2f>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	for (int i = 0; i < fileNamesL.size(); i++)
	{
		Mat img_left = imread(fileNamesL[i], IMREAD_GRAYSCALE);
		Mat img_right = imread(fileNamesL[i].substr(0, fileNamesL[i].length() - 5) + "R.jpg"
			, IMREAD_GRAYSCALE);

		if (img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (i == 0)
		{
			imgSize.width = img_left.cols;
			imgSize.height = img_left.rows;
		}

		//Mat resizeLeft;
		//Mat resizeRight;
		//resize(img_left, resizeLeft, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);
		//resize(img_right, resizeRight, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(img_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);//函数定位棋盘图案的内部棋盘角，若找到所有角，并且它们按特定顺序放置（逐行，每行从左到右），则函数返回非零值，否则返回0
		bool patternFound_right = findChessboardCorners(img_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternFound_left && patternFound_right)
		{
			cornerSubPix(img_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_left.push_back(cornerPts_left);
			drawChessboardCorners(img_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(img_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));
			cornerPtsVec_right.push_back(cornerPts_right);
			drawChessboardCorners(img_right, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3f>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	std::vector<Point3f> tempPts;
	//for (int i = patternSize.height - 1; i >= 0; i--)
	//{
	//	for (int j = 0; j < patternSize.width; j++)
	//	{
	//		Point3f realPt;
	//		realPt.x = j * squareSize.width;
	//		realPt.y = i * squareSize.height;
	//		realPt.z = 0;
	//		tempPts.push_back(realPt);
	//	}
	//}
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

	for (int i = 0; i < cornerPtsVec_right.size(); i++)
	{
		objPts3d.push_back(tempPts);
	}

	Mat cameraMatrixInnerPara_left = Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image
	std::vector<Mat> vectorMatR_left;										//matrix R of each image
	double rmsLeft = calibrateCamera(objPts3d, cornerPtsVec_left, imgSize,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		CALIB_FIX_PRINCIPAL_POINT
		| CALIB_RATIONAL_MODEL);// | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO 

	Mat cameraMatrixInnerPara_right = Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 5, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = calibrateCamera(objPts3d, cornerPtsVec_right, imgSize,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right,
		CALIB_FIX_PRINCIPAL_POINT
		| CALIB_RATIONAL_MODEL);//| CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat matrixE;				//essential matrix E
	Mat matrixF;				//fundamental matrix F
	double rms = stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
		imgSize, matrixR, matrixT, matrixE, matrixF, CALIB_FIX_INTRINSIC,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 50, 1e-6));

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << cameraMatrixInnerPara_left;
	fn << "Left_CameraDistPara" << cameraMatrixDistPara_left;
	fn << "Right_CameraInnerPara" << cameraMatrixInnerPara_right;
	fn << "Right_CameraDistPara" << cameraMatrixDistPara_right;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn << "EssentialMat" << matrixE;
	fn << "FundamentalMat" << matrixF;
	fn.release();

	//double err = 0;
	//int npoints = 0;
	//std::vector<Vec3f> lines[2];
	//for (int i = 0; i < cornerPtsVec_left.size(); i++)
	//{
	//	int npt = (int)cornerPtsVec_left[i].size();
	//	Mat imgpt[2];
	//	imgpt[0] = Mat(cornerPtsVec_left[i]);
	//	undistortPoints(imgpt[0], imgpt[0], cameraMatrixInnerPara_left, cameraMatrixDistPara_left, Mat(), cameraMatrixInnerPara_left);
	//	computeCorrespondEpilines(imgpt[0], 0 + 1, matrixF, lines[0]);

	//	imgpt[1] = Mat(cornerPtsVec_right[i]);
	//	undistortPoints(imgpt[1], imgpt[1], cameraMatrixInnerPara_right, cameraMatrixDistPara_right, Mat(), cameraMatrixInnerPara_right);
	//	computeCorrespondEpilines(imgpt[1], 1 + 1, matrixF, lines[1]);

	//	for (int j = 0; j < npt; j++)
	//	{
	//		double errij = fabs(cornerPtsVec_left[i][j].x*lines[1][j][0] +
	//			cornerPtsVec_left[i][j].y*lines[1][j][1] + lines[1][j][2]) +
	//			fabs(cornerPtsVec_right[i][j].x*lines[0][j][0] +
	//				cornerPtsVec_right[i][j].y*lines[0][j][1] + lines[0][j][2]);
	//		err += errij;
	//	}
	//	npoints += npt;
	//}
	//std::cout << "average epipolar err = " << err / npoints << std::endl;
}

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

void stereoFisheyeCamCalib(std::string imgFilePathL, std::string imgFilePathR, std::string cameraParaPath)
{
	//read the two cameras' pre-captured  images:left camera, right camera
//load all the images in the folder
	String filePathL = imgFilePathL + "/*L.jpg";
	std::vector<String> fileNamesL;
	glob(filePathL, fileNamesL, false);
	Size patternSize(9, 6);		//表示棋盘行和列的内角数（行，列）	//9:the number of inner corners in each row of the chess board
																					//6:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2d>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	for (int i = 0; i < fileNamesL.size(); i++)
	{
		Mat img_left = imread(fileNamesL[i], IMREAD_GRAYSCALE);
		Mat img_right = imread(fileNamesL[i].substr(0, fileNamesL[i].length() - 5) + "R.jpg"
			, IMREAD_GRAYSCALE);

		if (img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (i == 0)
		{
			imgSize.width = img_left.cols;
			imgSize.height = img_left.rows;
		}

		//Mat resizeLeft;
		//Mat resizeRight;
		//resize(img_left, resizeLeft, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);
		//resize(img_right, resizeRight, Size(imgSize.width * 2, imgSize.height * 2), 0.0, 0.0, INTER_LINEAR);

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(img_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);//函数定位棋盘图案的内部棋盘角，若找到所有角，并且它们按特定顺序放置（逐行，每行从左到右），则函数返回非零值，否则返回0
		bool patternFound_right = findChessboardCorners(img_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternFound_left && patternFound_right)
		{
			//获取角点更精细的检测结果
			cornerSubPix(img_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6));
			cornerPtsVec_left.push_back(VecPointF2D(cornerPts_left));
			drawChessboardCorners(img_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(img_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6));
			cornerPtsVec_right.push_back(VecPointF2D(cornerPts_right));
			drawChessboardCorners(img_right, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3d>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	std::vector<Point3d> tempPts;
	//for (int i = patternSize.height - 1; i >= 0; i--)
	//{
	//	for (int j = 0; j < patternSize.width; j++)
	//	{
	//		Point3f realPt;
	//		realPt.x = j * squareSize.width;
	//		realPt.y = i * squareSize.height;
	//		realPt.z = 0;
	//		tempPts.push_back(realPt);
	//	}
	//}
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3d realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}

	for (int i = 0; i < cornerPtsVec_right.size(); i++)
	{
		objPts3d.push_back(tempPts);
	}

	Mat cameraMatrixInnerPara_left = Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image:translation
	std::vector<Mat> vectorMatR_left;										//matrix R of each image:rotation
	double rmsLeft = fisheye::calibrate(objPts3d, cornerPtsVec_left, imgSize,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
		cv::TermCriteria(3, 20, 1e-6));// | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO 

	Mat cameraMatrixInnerPara_right = Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = fisheye::calibrate(objPts3d, cornerPtsVec_right, imgSize,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right,
		fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
		cv::TermCriteria(3, 20, 1e-6));//| CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat matrixE;				//essential matrix E
	Mat matrixF;				//fundamental matrix F

	double rms = fisheye::stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
		imgSize, matrixR, matrixT,
		fisheye::CALIB_USE_INTRINSIC_GUESS | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-12));

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << cameraMatrixInnerPara_left;
	fn << "Left_CameraDistPara" << cameraMatrixDistPara_left;
	fn << "Right_CameraInnerPara" << cameraMatrixInnerPara_right;
	fn << "Right_CameraDistPara" << cameraMatrixDistPara_right;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn << "EssentialMat" << matrixE;
	fn << "FundamentalMat" << matrixF;
	fn.release();

}

void stereoFisheyCamCalibRecti(std::string imgFilePathL, std::string cameraParaPath)
{
	//read the two cameras' pre-captured  images:left camera, right camera
	//load all the images in the folder
	String filePathL = imgFilePathL + "/*L.jpg";
	std::vector<String> fileNamesL;
	glob(filePathL, fileNamesL, false);
	Size patternSize(9, 6);		//表示棋盘行和列的内角数（行，列）	//9:the number of inner corners in each row of the chess board
																					//6:the number of inner corners in each col of the chess board
	int gridPatternNum = patternSize.width * patternSize.height;

	Size imgSize;

	//detect the inner corner in each chess image
	std::vector<std::vector<Point2d>> cornerPtsVec_left, cornerPtsVec_right;		//store the detected inner corners of each image
	for (int i = 0; i < fileNamesL.size(); i++)
	{
		Mat img_left = imread(fileNamesL[i], IMREAD_GRAYSCALE);
		Mat img_right = imread(fileNamesL[i].substr(0, fileNamesL[i].length() - 5) + "R.jpg"
			, IMREAD_GRAYSCALE);

		if (img_left.rows != img_right.rows && img_left.cols != img_right.cols)
		{
			std::cout << "img reading error" << std::endl;
			return;
		}

		if (i == 0)
		{
			imgSize.width = img_left.cols;
			imgSize.height = img_left.rows;
		}

		std::vector<Point2f> cornerPts_left, cornerPts_right;
		bool patternFound_left = findChessboardCorners(img_left, patternSize, cornerPts_left, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);//函数定位棋盘图案的内部棋盘角，若找到所有角，并且它们按特定顺序放置（逐行，每行从左到右），则函数返回非零值，否则返回0
		bool patternFound_right = findChessboardCorners(img_right, patternSize, cornerPts_right, CALIB_CB_ADAPTIVE_THRESH
			| CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (patternFound_left && patternFound_right)
		{
			//获取角点更精细的检测结果
			cornerSubPix(img_left, cornerPts_left, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6));
			cornerPtsVec_left.push_back(VecPointF2D(cornerPts_left));
			drawChessboardCorners(img_left, patternSize, cornerPts_left, patternFound_left);

			cornerSubPix(img_right, cornerPts_right, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6));
			cornerPtsVec_right.push_back(VecPointF2D(cornerPts_right));
			drawChessboardCorners(img_right, patternSize, cornerPts_right, patternFound_right);
		}
	}

	//two cameras calibration
	Size2f squareSize(100, 100);		//the real size of each grid in the chess board,which is measured manually by ruler

	std::vector<std::vector<Point3d>> objPts3d;					 	//calculated coordination of corners in world coordinate system
	std::vector<Point3d> tempPts;
	for (int i = 0; i < patternSize.height; i++)
	{
		for (int j = 0; j < patternSize.width; j++)
		{
			Point3d realPt;
			realPt.x = j * squareSize.width;
			realPt.y = i * squareSize.height;
			realPt.z = 0;
			tempPts.push_back(realPt);
		}
	}

	for (int i = 0; i < cornerPtsVec_right.size(); i++)
	{
		objPts3d.push_back(tempPts);
	}

	Mat cameraMatrixInnerPara_left = Mat::eye(3, 3, CV_64F);		//the inner parameters of camera
	Mat cameraMatrixDistPara_left = Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_left;										//matrix T of each image:translation
	std::vector<Mat> vectorMatR_left;										//matrix R of each image:rotation
	double rmsLeft = fisheye::calibrate(objPts3d, cornerPtsVec_left, imgSize,
		cameraMatrixInnerPara_left, cameraMatrixDistPara_left, vectorMatR_left, vectorMatT_left,
		fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
		cv::TermCriteria(3, 100, 1e-6));// | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO 

	Mat cameraMatrixInnerPara_right = Mat::eye(3, 3, CV_64F);	//the inner parameters of camera
	Mat cameraMatrixDistPara_right = Mat(1, 4, CV_64F, Scalar::all(0));		//the paramters of camera distortion
	std::vector<Mat> vectorMatT_right;									//matrix T of each image
	std::vector<Mat> vectorMatR_right;									//matrix R of each image
	double rmsRight = fisheye::calibrate(objPts3d, cornerPtsVec_right, imgSize,
		cameraMatrixInnerPara_right, cameraMatrixDistPara_right, vectorMatR_right, vectorMatT_right,
		fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW,
		cv::TermCriteria(3, 100, 1e-6));//| CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 | CALIB_FIX_ASPECT_RATIO

	cout << "rmsLeft:" << rmsLeft << endl;
	cout << "rmsRight:" << rmsRight << endl;

	//Mat monoMapL1, monoMapL2, monoMapR1, monoMapR2;
	////Precompute maps for cv::remap()
	//fisheye::initUndistortRectifyMap(cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
	//	noArray(), cameraMatrixInnerPara_left,
	//	imgSize, CV_32F, monoMapL1, monoMapL2);

	//fisheye::initUndistortRectifyMap(cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
	//	noArray(), cameraMatrixInnerPara_right,
	//	imgSize, CV_32F, monoMapR1, monoMapR2);


	//for (int i = 0; i < cornerPtsVec_left.size(); i++)
	//{
	//	fisheye::undistortPoints(cornerPtsVec_left[i], cornerPtsVec_left[i],
	//		cameraMatrixInnerPara_left, cameraMatrixDistPara_left,
	//		noArray(), cameraMatrixInnerPara_left);

	//	fisheye::undistortPoints(cornerPtsVec_right[i], cornerPtsVec_right[i],
	//		cameraMatrixInnerPara_right, cameraMatrixDistPara_right,
	//		noArray(), cameraMatrixInnerPara_right);
	//}

	Mat K1, D1, K2, D2;
	Mat matrixR, matrixT;		//the rotation mattrix and translate matrix of the right camera related to the left camera
	Mat zeroDistortion = Mat::zeros(cameraMatrixDistPara_right.size(), cameraMatrixDistPara_right.type());

	int flag = 0;
	flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flag |= cv::fisheye::CALIB_CHECK_COND;
	flag |= cv::fisheye::CALIB_FIX_SKEW;
	double rms = fisheye::stereoCalibrate(objPts3d, cornerPtsVec_left, cornerPtsVec_right,
		cameraMatrixInnerPara_left, zeroDistortion,
		cameraMatrixInnerPara_right, zeroDistortion,
		//K1, D1, K2, D2,
		imgSize, matrixR, matrixT,
		flag);

	std::cout << "stereo_calibration_error" << rms << std::endl;

	FileStorage fn(cameraParaPath, FileStorage::WRITE);
	fn << "ImgSize" << imgSize;
	fn << "Left_CameraInnerPara" << K1;
	fn << "Left_CameraDistPara" << D1;
	fn << "Right_CameraInnerPara" << K2;
	fn << "Right_CameraDistPara" << D2;
	fn << "R2L_Rotation_Matrix" << matrixR;
	fn << "R2L_Translate_Matrix" << matrixT;
	fn.release();

}

/**
 * \brief using pre-calibrated parameters to rectify the images captured by left and right cameras in a real-time manner
 * \param cameraParaPath :the .xml file path of the pre-calculated camera calibration parameters
 */
void stereoCameraUndistort(std::string cameraParaPath)
{
	Size imgSize;
	Mat cameraInnerPara_left, cameraInnerPara_right;
	Mat cameraDistPara_left, cameraDistPara_right;
	Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_Matrix"] >> matrixR;
	fn["R2L_Translate_Matrix"] >> matrixT;
	fn.release();


	Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q;
	Rect validRoi[2];

	stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q,
		CALIB_ZERO_DISPARITY, 0.0, imgSize, &validRoi[0], &validRoi[1]);

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left, matrixRectify_left, matrixProjection_left, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right, matrixRectify_right, matrixProjection_right, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;  //检验校正是否正确
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3); //高度一样，宽度双倍
	}
	else
	{
		sf = 300. / MAX(imgSize.width, imgSize.height);
		w = cvRound(imgSize.width*sf);
		h = cvRound(imgSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3); //高度双倍，宽度一样
	}

	destroyAllWindows();

	VideoCapture cap_left, cap_right;
	cap_left.open(0);
	if (!cap_left.isOpened())
	{
		std::cout << "Left camera open failed!" << std::endl;
		return;
	}
	cap_left.set(CAP_PROP_FOURCC, 'GPJM');
	cap_left.set(CAP_PROP_FRAME_HEIGHT, imgSize.height);		//rows
	cap_left.set(CAP_PROP_FRAME_WIDTH, imgSize.width);			//cols

	cap_right.open(1);
	if (!cap_right.isOpened())
	{
		std::cout << "Left camera open failed!" << std::endl;
		return;
	}
	cap_right.set(CAP_PROP_FOURCC, 'GPJM');
	cap_right.set(CAP_PROP_FRAME_HEIGHT, cap_left.get(CAP_PROP_FRAME_HEIGHT));
	cap_right.set(CAP_PROP_FRAME_WIDTH, cap_left.get(CAP_PROP_FRAME_WIDTH));



	int ndisparities = 16 * 5;   /**< Range of disparity */
	int SADWindowSize = 31; /**< Size of the block window. Must be odd */
	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
	sbm->setMinDisparity(0);
	//sbm->setNumDisparities(64);
	sbm->setTextureThreshold(10);
	sbm->setDisp12MaxDiff(-1);
	sbm->setPreFilterCap(31);
	sbm->setUniquenessRatio(25);
	sbm->setSpeckleRange(32);
	sbm->setSpeckleWindowSize(100);


	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 64, 7,
		10 * 7 * 7,
		40 * 7 * 7,
		1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);

	Mat frame_left, frame_right;
	Mat imgLeft, imgRight;
	Mat rimg, cimg;
	Mat Mask;
	while (true)
	{
		cap_left >> frame_left;
		cap_right >> frame_right;

		if (frame_left.empty() || frame_right.empty())
			continue;

		remap(frame_left, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h)); //浅拷贝
		resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA); //INTER_AREA：插值方法
		Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
			cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

		remap(frame_right, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h)); //浅拷贝
		resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
		Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
			cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

		Rect vroi = vroi1 & vroi2;		//all pixels in the roi are valid in both left and right view maps

		imgLeft = canvasPart1(vroi).clone();  //rectified images with only valid pixels
		imgRight = canvasPart2(vroi).clone();

		//draw the valid rectangle in two views respectively
		rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
		rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

		//draw lines for verification quality evaluation
		if (!isVerticalStereo)
			for (int j = 0; j < canvas.rows; j += 32)
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (int j = 0; j < canvas.cols; j += 32)
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

		//the input image for stereo matching is supposed to be 8-bit single channel image
		cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
		cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);

		//-- And create the image in which we will save our disparities
		Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S); //for BM
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S); //for SGBM
		Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl;
			return;
		}

		sbm->compute(imgLeft, imgRight, imgDisparity16S);

		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
		applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
		Mat disparityShow;
		imgDisparity8U.copyTo(disparityShow, Mask);




		sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

		sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
		applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
		Mat  sgbmDisparityShow;
		sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

		imshow("bmDisparity", disparityShow);
		imshow("sgbmDisparity", sgbmDisparityShow);
		imshow("rectified", canvas);
		char c = (char)waitKey(1);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

/**
 * \brief using pre-calibrated parameters to rectify the images pre-captured by left and right cameras
 * \param imgFilePath
 * \param cameraParaPath
 */
void stereoCameraUndistort(std::string imgFilePath, std::string cameraParaPath)
{
	Size imgSize;
	Mat cameraInnerPara_left, cameraInnerPara_right;
	Mat cameraDistPara_left, cameraDistPara_right;
	Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_Matrix"] >> matrixR;
	fn["R2L_Translate_Matrix"] >> matrixT;
	fn.release();

	Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right;
	Rect validRoi[2];

	Mat R_L, R_R, P_L, P_R, Q;
	fisheye::stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, R_L, R_R, P_L, P_R, Q,
		CALIB_ZERO_DISPARITY, imgSize, 0.0, 1.1);

	//// OpenCV can handle left-right
	//// or up-down camera arrangements
	//bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	Mat lmapx, lmapy, rmapx, rmapy;
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	fisheye::initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left,
		R_L, P_L,
		imgSize * 2, CV_32F, lmapx, lmapy);

	fisheye::initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right,
		R_R, P_R,
		imgSize * 2, CV_32F, rmapx, rmapy);

	//Mat canvas;
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
	int SADWindowSize = 31; /**< Size of the block window. Must be odd */
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

	Mat frame_left, frame_right;
	Mat imgLeft, imgRight;
	Mat rimg, cimg;
	Mat Mask;

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

		Mat lundist, rundist;
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

		//Mat imgLeftBGR = imgLeft.clone();
		//Mat imgRightBGR = imgRight.clone();

		//cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
		//cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);


		////-- And create the image in which we will save our disparities
		//Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		//Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		//Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		//Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		//if (imgLeft.empty() || imgRight.empty())
		//{
		//	std::cout << " --(!) Error reading images " << std::endl;
		//	return;
		//}

		//sbm->compute(imgLeft, imgRight, imgDisparity16S);

		//imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
		//cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
		//applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
		//Mat disparityShow;
		//imgDisparity8U.copyTo(disparityShow, Mask);


		//sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

		//sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
		//cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
		//applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
		//Mat  sgbmDisparityShow;
		//sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

		//imshow("bmDisparity", disparityShow);
		//imshow("sgbmDisparity", sgbmDisparityShow);
		//imshow("rectified", canvas);

		//showPointCloud(imgLeftBGR, imgDisparity16S, cameraParaPath);

		char c = (char)waitKey(0);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

void getRectifiedImages(std::string imgFilePath, std::string cameraParaPath)
{
	Size imgSize;
	Mat cameraInnerPara_left, cameraInnerPara_right;
	Mat cameraDistPara_left, cameraDistPara_right;
	Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_Matrix"] >> matrixR;
	fn["R2L_Translate_Matrix"] >> matrixT;
	fn.release();


	Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q;
	Rect validRoi[2];

	stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q,
		CALIB_ZERO_DISPARITY, 0, imgSize, &validRoi[0], &validRoi[1]);

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left, matrixRectify_left, matrixProjection_left, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right, matrixRectify_right, matrixProjection_right, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
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
	int SADWindowSize = 31; /**< Size of the block window. Must be odd */
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

	Mat frame_left, frame_right;
	Mat imgLeft, imgRight;
	Mat rimg, cimg;
	Mat Mask;

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
		Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
		resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
		Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
			cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

		remap(frame_right, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
		rimg.copyTo(cimg);
		Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));
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

		Mat imgLeftBGR = imgLeft.clone();
		Mat imgRightBGR = imgRight.clone();

		cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
		cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);


		//-- And create the image in which we will save our disparities
		Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		if (imgLeft.empty() || imgRight.empty())
		{
			std::cout << " --(!) Error reading images " << std::endl;
			return;
		}

		sbm->compute(imgLeft, imgRight, imgDisparity16S);

		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
		applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
		Mat disparityShow;
		imgDisparity8U.copyTo(disparityShow, Mask);


		sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

		sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
		cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
		applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
		Mat  sgbmDisparityShow;
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
	Size imgSize;
	Mat cameraInnerPara_left, cameraInnerPara_right;
	Mat cameraDistPara_left, cameraDistPara_right;
	Mat matrixR, matrixT;
	FileStorage fn(cameraParaPath, FileStorage::READ);
	fn["ImgSize"] >> imgSize;
	fn["Left_CameraInnerPara"] >> cameraInnerPara_left;
	fn["Left_CameraDistPara"] >> cameraDistPara_left;
	fn["Right_CameraInnerPara"] >> cameraInnerPara_right;
	fn["Right_CameraDistPara"] >> cameraDistPara_right;
	fn["R2L_Rotation_Matrix"] >> matrixR;
	fn["R2L_Translate_Matrix"] >> matrixT;
	fn.release();


	Mat matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q;
	Rect validRoi[2];

	stereoRectify(cameraInnerPara_left, cameraDistPara_left,
		cameraInnerPara_right, cameraDistPara_right,
		imgSize, matrixR, matrixT, matrixRectify_left, matrixRectify_right, matrixProjection_left, matrixProjection_right, Q,
		CALIB_ZERO_DISPARITY, 0, imgSize, &validRoi[0], &validRoi[1]);

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(matrixProjection_right.at<double>(1, 3)) > fabs(matrixProjection_right.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraInnerPara_left, cameraDistPara_left, matrixRectify_left, matrixProjection_left, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraInnerPara_right, cameraDistPara_right, matrixRectify_right, matrixProjection_right, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
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

	Mat rimg, cimg;
	Mat Mask;

	if (imgLeft.rows != imgRight.rows && imgLeft.cols != imgRight.cols && imgLeft.rows != imgSize.height && imgLeft.cols != imgSize.width)
	{
		std::cout << "img reading error" << std::endl;
		return;
	}

	if (imgLeft.empty() || imgRight.empty())
		return;

	remap(imgLeft, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
	rimg.copyTo(cimg);
	Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
	resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
	Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
		cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

	remap(imgRight, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
	rimg.copyTo(cimg);
	Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));
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