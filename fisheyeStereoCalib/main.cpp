#include "methods/fisheyeCalib3d.h"

using namespace cv;
using namespace std;

int main()
{
	std::string imgFilePath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-08";
	std::string xmlFilePath = "D:\\studying\\stereo vision\\research code\\data\\2019-07-08\\stereoCalibrateData20190708.xml";
	stereoFisheyCamCalibRecti(imgFilePath, xmlFilePath);

	return 0;
}