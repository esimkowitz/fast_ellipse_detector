/*
This code is intended for academic use only.
You are free to use and modify the code, at your own risk.

If you use this code, or find it useful, please refer to the paper:

Michele Fornaciari, Andrea Prati, Rita Cucchiara,
A fast and effective ellipse detector for embedded vision applications
Pattern Recognition, Volume 47, Issue 11, November 2014, Pages 3693-3708, ISSN 0031-3203,
http://dx.doi.org/10.1016/j.patcog.2014.05.012.
(http://www.sciencedirect.com/science/article/pii/S0031320314001976)


The comments in the code refer to the abovementioned paper.
If you need further details about the code or the algorithm, please contact me at:

michele.fornaciari@unimore.it

last update: 23/12/2014
*/

//#include <cv.hpp>
#include <limits.h> /* PATH_MAX */
#include <stdlib.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>

#include "EllipseDetectorYaed.h"
#include <fstream>

using namespace std;
using namespace cv;

// Should be checked
void SaveEllipses(const string &workingDir, const string &imgName, const vector<Ellipse> &ellipses /*, const vector<double>& times*/)
{
	string path(workingDir + "/" + imgName + ".txt");
	ofstream out(path, ofstream::out | ofstream::trunc);
	if (!out.good())
	{
		cout << "Error saving: " << path << endl;
		return;
	}

	// Save execution time
	//out << times[0] << "\t" << times[1] << "\t" << times[2] << "\t" << times[3] << "\t" << times[4] << "\t" << times[5] << "\t" << "\n";

	unsigned n = ellipses.size();
	// Save number of ellipses
	out << n << "\n";

	// Save ellipses
	for (unsigned i = 0; i < n; ++i)
	{
		const Ellipse &e = ellipses[i];
		out << e._xc << "\t" << e._yc << "\t" << e._a << "\t" << e._b << "\t" << e._rad << "\t" << e._score << "\n";
	}
	out.close();
}

// Should be checked
bool LoadTest(vector<Ellipse> &ellipses, const string &sTestFileName, vector<double> &times, bool bIsAngleInRadians = true)
{
	ifstream in(sTestFileName);
	if (!in.good())
	{
		cout << "Error opening: " << sTestFileName << endl;
		return false;
	}

	times.resize(6);
	in >> times[0] >> times[1] >> times[2] >> times[3] >> times[4] >> times[5];

	unsigned n;
	in >> n;

	ellipses.clear();

	if (n == 0)
		return true;

	ellipses.reserve(n);

	while (in.good() && n--)
	{
		Ellipse e;
		in >> e._xc >> e._yc >> e._a >> e._b >> e._rad >> e._score;

		if (!bIsAngleInRadians)
		{
			e._rad = e._rad * float(CV_PI / 180.0);
		}

		e._rad = fmod(float(e._rad + 2.0 * CV_PI), float(CV_PI));

		if ((e._a > 0) && (e._b > 0) && (e._rad >= 0))
		{
			ellipses.push_back(e);
		}
	}
	in.close();

	// Sort ellipses by decreasing score
	sort(ellipses.begin(), ellipses.end());

	return true;
}

void LoadGT(vector<Ellipse> &gt, const string &sGtFileName, bool bIsAngleInRadians = true)
{
	ifstream in(sGtFileName);
	if (!in.good())
	{
		cout << "Error opening: " << sGtFileName << endl;
		return;
	}

	unsigned n;
	in >> n;

	gt.clear();
	gt.reserve(n);

	while (in.good() && n--)
	{
		Ellipse e;
		in >> e._xc >> e._yc >> e._a >> e._b >> e._rad;

		if (!bIsAngleInRadians)
		{
			// convert to radians
			e._rad = float(e._rad * CV_PI / 180.0);
		}

		if (e._a < e._b)
		{
			float temp = e._a;
			e._a = e._b;
			e._b = temp;

			e._rad = e._rad + float(0.5 * CV_PI);
		}

		e._rad = fmod(float(e._rad + 2.f * CV_PI), float(CV_PI));
		e._score = 1.f;
		gt.push_back(e);
	}
	in.close();
}

bool TestOverlap(const Mat1b &gt, const Mat1b &test, float th)
{
	float fAND = float(countNonZero(gt & test));
	float fOR = float(countNonZero(gt | test));
	float fsim = fAND / fOR;

	return (fsim >= th);
}

int Count(const vector<bool> v)
{
	int counter = 0;
	for (unsigned i = 0; i < v.size(); ++i)
	{
		if (v[i])
		{
			++counter;
		}
	}
	return counter;
}

// Should be checked !!!!!
std::tuple<float, float, float> Evaluate(const vector<Ellipse> &ellGT, const vector<Ellipse> &ellTest, const float th_score, const Mat3b &img)
{
	float threshold_overlap = 0.8f;
	//float threshold = 0.95f;

	unsigned sz_gt = ellGT.size();
	unsigned size_test = ellTest.size();

	unsigned sz_test = unsigned(min(1000, int(size_test)));

	vector<Mat1b> gts(sz_gt);
	vector<Mat1b> tests(sz_test);

	for (unsigned i = 0; i < sz_gt; ++i)
	{
		const Ellipse &e = ellGT[i];

		Mat1b tmp(img.rows, img.cols, uchar(0));
		ellipse(tmp, Point(e._xc, e._yc), Size(e._a, e._b), e._rad * 180.0 / CV_PI, 0.0, 360.0, Scalar(255), -1);
		gts[i] = tmp;
	}

	for (unsigned i = 0; i < sz_test; ++i)
	{
		const Ellipse &e = ellTest[i];

		Mat1b tmp(img.rows, img.cols, uchar(0));
		ellipse(tmp, Point(e._xc, e._yc), Size(e._a, e._b), e._rad * 180.0 / CV_PI, 0.0, 360.0, Scalar(255), -1);
		tests[i] = tmp;
	}

	Mat1b overlap(sz_gt, sz_test, uchar(0));
	for (int r = 0; r < overlap.rows; ++r)
	{
		for (int c = 0; c < overlap.cols; ++c)
		{
			overlap(r, c) = TestOverlap(gts[r], tests[c], threshold_overlap) ? uchar(255) : uchar(0);
		}
	}

	int counter = 0;

	vector<bool> vec_gt(sz_gt, false);

	for (int i = 0; i < sz_test; ++i)
	{
		const Ellipse &e = ellTest[i];
		for (int j = 0; j < sz_gt; ++j)
		{
			if (vec_gt[j])
			{
				continue;
			}

			bool bTest = overlap(j, i) != 0;

			if (bTest)
			{
				vec_gt[j] = true;
				break;
			}
		}
	}

	int tp = Count(vec_gt);
	int fn = int(sz_gt) - tp;
	int fp = size_test - tp; // !!!!

	float pr(0.f);
	float re(0.f);
	float fmeasure(0.f);

	if (tp == 0)
	{
		if (fp == 0)
		{
			pr = 1.f;
			re = 0.f;
			fmeasure = (2.f * pr * re) / (pr + re);
		}
		else
		{
			pr = 0.f;
			re = 0.f;
			fmeasure = 0.f;
		}
	}
	else
	{
		pr = float(tp) / float(tp + fp);
		re = float(tp) / float(tp + fn);
		fmeasure = (2.f * pr * re) / (pr + re);
	}

	return make_tuple(pr, re, fmeasure);
}

vector<Point> getQuadrilateral(Mat & grayscale, Mat& output)
{
    Mat approxPoly_mask(grayscale.rows, grayscale.cols, CV_8UC1);
    approxPoly_mask = Scalar(0);

    vector<vector<Point>> contours;
    findContours(grayscale, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    vector<int> indices(contours.size());
    iota(indices.begin(), indices.end(), 0);

    sort(indices.begin(), indices.end(), [&contours](int lhs, int rhs) {
        return contours[lhs].size() > contours[rhs].size();
    });

    /// Find the convex hull object for each contour
    vector<vector<Point> >hull(1);
    convexHull(Mat(contours[indices[0]]), hull[0], false);

    vector<vector<Point>> polygon(1);
    approxPolyDP(hull[0], polygon[0], 20, true);
    drawContours(approxPoly_mask, polygon, 0, Scalar(255));
	namedWindow("approxPoly_mask", WINDOW_NORMAL);
    imshow("approxPoly_mask", approxPoly_mask);

    if (polygon[0].size() >= 4) // we found the 4 corners
    {
        return(polygon[0]);
    }

    return(vector<Point>());
}

void detect_quadrilateral(char *image_path) {
	// Check if the file provided is a valid image
	string filename(image_path);
	string file_basename = basename(image_path);
	string ext = file_basename.substr(file_basename.find_last_of(".") + 1);
	if (!((ext == "jpeg") || (ext == "jpg")))
	{
		cout << "image must be .jpeg or .jpg" << endl;
		return;
	}

	cout << "detect quadrilateral for image \"" << image_path << "\"" << endl;
	Mat input = imread(filename);
	resize(input, input, Size(), 0.1, 0.1);
    Mat input_grey;
    cvtColor(input, input_grey, CV_BGR2GRAY);
    Mat threshold1;
    Mat edges;
    blur(input_grey, input_grey, Size(3, 3));
    Canny(input_grey, edges, 30, 100);


    vector<Point> card_corners = getQuadrilateral(edges, input);
    Mat warpedCard(400, 300, CV_8UC3);
    if (card_corners.size() == 4)
    {
        Mat homography = findHomography(card_corners, vector<Point>{Point(warpedCard.cols, warpedCard.rows), Point(0, warpedCard.rows), Point(0, 0), Point(warpedCard.cols, 0)});
        warpPerspective(input, warpedCard, homography, Size(warpedCard.cols, warpedCard.rows));
    }

    imshow("warped card", warpedCard);
    imshow("edges", edges);
    imshow("input", input);
    waitKey(0);
}

void detect_lines_by_color(char *image_path)
{ // Check if the file provided is a valid image
	string filename(image_path);
	string file_basename = basename(image_path);
	string ext = file_basename.substr(file_basename.find_last_of(".") + 1);
	if (!((ext == "jpeg") || (ext == "jpg")))
	{
		cout << "image must be .jpeg or .jpg" << endl;
		return;
	}

	string filename_minus_ext = filename.substr(0, filename.find_last_of("."));
	cout << "detect lines for image \"" << image_path << "\"" << endl;
	cv::Mat input = cv::imread(filename);
	Mat gray;
	cvtColor(input, gray, CV_BGR2GRAY);
	Mat inverted = ~input;

	int erosion_size = 10;
	Mat element = getStructuringElement( MORPH_ELLIPSE,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
	Mat eroded;
	erode(inverted, eroded, element );
	namedWindow("eroded", WINDOW_NORMAL);
	imshow("eroded", eroded);
	cv::waitKey(0);
	cv::destroyAllWindows();

	#if 1
	//convert to HSV color space
	cv::Mat hsvImage;
	cv::cvtColor(eroded, hsvImage, CV_BGR2HSV);

	// split the channels
	std::vector<cv::Mat> hsvChannels;
	cv::split(hsvImage, hsvChannels);
	cv::namedWindow("hue", WINDOW_NORMAL);
	cv::namedWindow("saturation", WINDOW_NORMAL);
	cv::namedWindow("value", WINDOW_NORMAL);
	cv::imshow("hue", hsvChannels[0]);
	cv::imshow("saturation", hsvChannels[1]);
	cv::imshow("value", hsvChannels[2]);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// hue channels tells you the color tone, if saturation and value aren't too low.

	// red color is a special case, because the hue space is circular and red is exactly at the beginning/end of the circle.
	// in literature, hue space goes from 0 to 360 degrees, but OpenCV rescales the range to 0 up to 180, because 360 does not fit in a single byte. Alternatively there is another mode where 0..360 is rescaled to 0..255 but this isn't as common.
	int minHue = 0;  // red color
	int maxHue = 15; // how much difference from the desired color we want to include to the result If you increase this value, for example a red color would detect some orange values, too.

	int minSaturation = 0; // I'm not sure which value is good here...
	int maxSaturation = 40;
	int minValue = 100;		// not sure whether 50 is a good min value here...
	int maxValue = 255;

	// cv::Mat hueImage = hsvChannels[0]; // [hue, saturation, value]

	// is the color within the lower hue range?
	cv::Mat hueMask;
	cv::inRange(hsvChannels[0], minHue, maxHue, hueMask);

	// // if the desired color is near the border of the hue space, check the other side too:
	// // TODO: this won't work if "hueValue + hueRange > 180" - maybe use two different if-cases instead... with int lowerHueValue = hueValue - 180
	// if (hueValue - hueRange < 0 || hueValue + hueRange > 180)
	// {
	// 	cv::Mat hueMaskUpper;
	// 	int upperHueValue = hueValue + 180; // in reality this would be + 360 instead
	// 	cv::inRange(hueImage, upperHueValue - hueRange, upperHueValue + hueRange, hueMaskUpper);

	// 	// add this mask to the other one
	// 	hueMask = hueMask | hueMaskUpper;
	// }

	// now we have to filter out all the pixels where saturation and value do not fit the limits:
	// cv::Mat saturationMask = hsvChannels[1] > minSaturation && hsvChannels[1] < maxSaturation;
	cv::Mat valueMask;
	cv::inRange(hsvChannels[2], minValue, maxValue, valueMask);
	cv::Mat saturationMask;
	cv::inRange(hsvChannels[1], minSaturation, maxSaturation, saturationMask);
	cv::Mat resultMask = valueMask & saturationMask & hueMask;
	// cv::inRange(gray, 190, 255, resultMask);
	#endif
	#if 1
	vector<Vec4i> lines;
	HoughLinesP(resultMask, lines, 1, CV_PI / 180, 200, 500, 80);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(input, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}
	
	#endif
	#if 0
	vector<Vec2f> lines;
    HoughLines( resultMask, lines, 1, CV_PI/180, 200 );

    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        Point pt1(cvRound(x0 + 1000*(-b)),
                  cvRound(y0 + 1000*(a)));
        Point pt2(cvRound(x0 - 1000*(-b)),
                  cvRound(y0 - 1000*(a)));
        line( input, pt1, pt2, Scalar(0,0,255), 3, 8 );
    }
	#endif
	#if 0

	// vector<Point> card_corners = getQuadrilateral(resultMask, input);

	// Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
	// Mat morph;
	// morphologyEx(resultMask, morph, CV_MOP_CLOSE, kernel);

	int rectIdx = 0;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(eroded, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (size_t idx = 0; idx < contours.size(); idx++)
	{
		RotatedRect rect = minAreaRect(contours[idx]);
		double areaRatio = abs(contourArea(contours[idx])) / (rect.size.width * rect.size.height);
		if (areaRatio > .95)
		{
			rectIdx = idx;
			break;
		}
	}
	// get the convexhull of the contour
	vector<Point> hull;
	convexHull(contours[rectIdx], hull, false, true);

	// visualization
	Mat rgb;
	cvtColor(eroded, rgb, CV_GRAY2BGR);
	drawContours(rgb, contours, rectIdx, Scalar(0, 0, 255), 2);
	for(size_t i = 0; i < hull.size(); i++)
	{
		line(rgb, hull[i], hull[(i + 1)%hull.size()], Scalar(0, 255, 0), 2);
	}
	#endif
	cv::namedWindow("desired color", WINDOW_NORMAL);
	cv::imshow("desired color", resultMask);
	
	namedWindow("detected lines", WINDOW_NORMAL);
	imshow("detected lines", input);
	cv::waitKey(0);
}

void create_skeleton(char *image_path)
{
	// Check if the file provided is a valid image
	string filename(image_path);
	string file_basename = basename(image_path);
	string ext = file_basename.substr(file_basename.find_last_of(".") + 1);
	if (!((ext == "jpeg") || (ext == "jpg")))
	{
		cout << "image must be .jpeg or .jpg" << endl;
		return;
	}

	string filename_minus_ext = filename.substr(0, filename.find_last_of("."));
	cout << "Creating skeleton for image \"" << image_path << "\"" << endl;

	// Read image
	Mat3b image = imread(filename);

	// Convert to grayscale
	Mat1b gray;
	cvtColor(image, gray, CV_BGR2GRAY);

	cv::threshold(gray, gray, 127, 255, cv::THRESH_BINARY);
	cv::Mat skel(gray.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;
	do
	{
		cv::erode(gray, eroded, element);
		cv::dilate(eroded, temp, element); // temp = open(img)
		cv::subtract(gray, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(gray);

		done = (cv::countNonZero(gray) == 0);
	} while (!done);

	// Mat cdst;
	// cvtColor(skel, cdst, CV_GRAY2BGR);

	vector<Vec4i> lines;
	HoughLinesP(skel, lines, 1, CV_PI / 180, 90, 200, 25);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}
	namedWindow("Skeleton", WINDOW_NORMAL);
	cv::imshow("Skeleton", skel);
	namedWindow("detected lines", WINDOW_NORMAL);
	imshow("detected lines", image);
	cv::waitKey(0);
}

void OnImage(char *image_path)
{
	// Check if the file provided is a valid image
	string filename(image_path);
	string file_basename = basename(image_path);
	string ext = file_basename.substr(file_basename.find_last_of(".") + 1);
	if (!((ext == "jpeg") || (ext == "jpg")))
	{
		cout << "image must be .jpeg or .jpg" << endl;
		return;
	}

	string filename_minus_ext = filename.substr(0, filename.find_last_of("."));
	cout << "Annotating image \"" << image_path << "\"" << endl;

	// Read image
	Mat3b image = imread(filename);
	// resize(image, image, Size(), 0.8, 0.8, INTER_CUBIC);

	Size sz = image.size();

	// Convert to grayscale
	Mat1b gray;
	cvtColor(image, gray, CV_BGR2GRAY);

	// Parameters Settings (Sect. 4.2)
	int iThLength = 16;
	float fThObb = 3.0f;
	float fThPos = 1.0f;
	float fTaoCenters = 0.05f;
	int iNs = 16;
	float fMaxCenterDistance = sqrt(float(sz.width * sz.width + sz.height * sz.height)) * fTaoCenters;

	float fThScoreScore = 0.4f;

	// Other constant parameters settings.

	// Gaussian filter parameters, in pre-processing
	Size szPreProcessingGaussKernelSize = Size(5, 5);
	double dPreProcessingGaussSigma = 1.0;

	float fDistanceToEllipseContour = 0.1f; // (Sect. 3.3.1 - Validation)
	float fMinReliability = 0.4f;			// Const parameters to discard bad ellipses

	// Initialize Detector with selected parameters
	CEllipseDetectorYaed *yaed = new CEllipseDetectorYaed();
	yaed->SetParameters(szPreProcessingGaussKernelSize,
						dPreProcessingGaussSigma,
						fThPos,
						fMaxCenterDistance,
						iThLength,
						fThObb,
						fDistanceToEllipseContour,
						fThScoreScore,
						fMinReliability,
						iNs);

	// Detect
	vector<Ellipse> ellsYaed;
	Mat1b gray2 = gray.clone();
	yaed->Detect(gray2, ellsYaed);

	vector<double> times = yaed->GetTimes();
	cout << "--------------------------------" << endl;
	cout << "Execution Time: " << endl;
	cout << "Edge Detection: \t" << times[0] << endl;
	cout << "Pre processing: \t" << times[1] << endl;
	cout << "Grouping:       \t" << times[2] << endl;
	cout << "Estimation:     \t" << times[3] << endl;
	cout << "Validation:     \t" << times[4] << endl;
	cout << "Clustering:     \t" << times[5] << endl;
	cout << "--------------------------------" << endl;
	cout << "Total:	         \t" << yaed->GetExecTime() << endl;
	cout << "--------------------------------" << endl;

	vector<Ellipse> gt;
	LoadGT(gt, filename_minus_ext + ".txt", true); // Prasad is in radians

	Mat3b resultImage = image.clone();

	// Draw GT ellipses
	for (unsigned i = 0; i < gt.size(); ++i)
	{
		Ellipse &e = gt[i];
		Scalar color(0, 0, 255);
		ellipse(resultImage, Point(cvRound(e._xc), cvRound(e._yc)), Size(cvRound(e._a), cvRound(e._b)), e._rad * 180.0 / CV_PI, 0.0, 360.0, color, 3);
	}

	yaed->DrawDetectedEllipses(resultImage, ellsYaed);

	Mat3b res = image.clone();

	Evaluate(gt, ellsYaed, fThScoreScore, res);

	// Show the image in a scalable window.
	namedWindow("Annotated Image", WINDOW_NORMAL);
	imshow("Annotated Image", resultImage);
	waitKey();
}

void OnVideo()
{

	string sWorkingDir = "/home/itv/Desktop/ellipse_detect";
	string imagename = "1.jpg";

	string filename = sWorkingDir + "/images/" + imagename;

	VideoCapture cap(0);
	if (!cap.isOpened())
		return;

	int width = 800;
	int height = 600;

	// Parameters Settings (Sect. 4.2)
	int iThLength = 16;
	float fThObb = 3.0f;
	float fThPos = 1.0f;
	float fTaoCenters = 0.05f;
	int iNs = 16;
	float fMaxCenterDistance = sqrt(float(width * width + height * height)) * fTaoCenters;

	float fThScoreScore = 0.4f;

	// Other constant parameters settings.

	// Gaussian filter parameters, in pre-processing
	Size szPreProcessingGaussKernelSize = Size(5, 5);
	double dPreProcessingGaussSigma = 1.0;

	float fDistanceToEllipseContour = 0.1f; // (Sect. 3.3.1 - Validation)
	float fMinReliability = 0.4f;			// Const parameters to discard bad ellipses

	// Initialize Detector with selected parameters
	CEllipseDetectorYaed *yaed = new CEllipseDetectorYaed();
	yaed->SetParameters(szPreProcessingGaussKernelSize,
						dPreProcessingGaussSigma,
						fThPos,
						fMaxCenterDistance,
						iThLength,
						fThObb,
						fDistanceToEllipseContour,
						fThScoreScore,
						fMinReliability,
						iNs);

	Mat1b gray;
	while (true)
	{
		Mat3b image;
		cap >> image;
		cvtColor(image, gray, CV_BGR2GRAY);

		vector<Ellipse> ellsYaed;
		Mat1b gray2 = gray.clone();
		yaed->Detect(gray2, ellsYaed);

		vector<double> times = yaed->GetTimes();
		cout << "--------------------------------" << endl;
		cout << "Execution Time: " << endl;
		cout << "Edge Detection: \t" << times[0] << endl;
		cout << "Pre processing: \t" << times[1] << endl;
		cout << "Grouping:       \t" << times[2] << endl;
		cout << "Estimation:     \t" << times[3] << endl;
		cout << "Validation:     \t" << times[4] << endl;
		cout << "Clustering:     \t" << times[5] << endl;
		cout << "--------------------------------" << endl;
		cout << "Total:	         \t" << yaed->GetExecTime() << endl;
		cout << "--------------------------------" << endl;

		vector<Ellipse> gt;
		LoadGT(gt, sWorkingDir + "/gt/" + "gt_" + imagename + ".txt", true); // Prasad is in radians

		Mat3b resultImage = image.clone();

		// Draw GT ellipses
		for (unsigned i = 0; i < gt.size(); ++i)
		{
			Ellipse &e = gt[i];
			Scalar color(0, 0, 255);
			ellipse(resultImage, Point(cvRound(e._xc), cvRound(e._yc)), Size(cvRound(e._a), cvRound(e._b)), e._rad * 180.0 / CV_PI, 0.0, 360.0, color, 3);
		}

		yaed->DrawDetectedEllipses(resultImage, ellsYaed);

		Mat3b res = image.clone();

		Evaluate(gt, ellsYaed, fThScoreScore, res);

		imshow("Yaed", resultImage);

		if (waitKey(10) >= 0)
			break;
	}
}

void OnDataset()
{
	string sWorkingDir = "D:\\data\\ellipse_dataset\\Random Images - Dataset #1\\";
	//string sWorkingDir = "D:\\data\\ellipse_dataset\\Prasad Images - Dataset Prasad\\";
	string out_folder = "D:\\data\\ellipse_dataset\\";

	vector<string> names;

	vector<float> prs;
	vector<float> res;
	vector<float> fms;
	vector<double> tms;

	glob(sWorkingDir + "images\\" + "*.*", names);

	int counter = 0;
	for (const auto &image_name : names)
	{
		cout << double(counter++) / names.size() << "\n";

		string name_ext = image_name.substr(image_name.find_last_of("\\") + 1);
		string name = name_ext.substr(0, name_ext.find_last_of("."));

		Mat3b image = imread(image_name);
		Size sz = image.size();

		// Convert to grayscale
		Mat1b gray;
		cvtColor(image, gray, CV_BGR2GRAY);

		// Parameters Settings (Sect. 4.2)
		int iThLength = 16;
		float fThObb = 3.0f;
		float fThPos = 1.0f;
		float fTaoCenters = 0.05f;
		int iNs = 16;
		float fMaxCenterDistance = sqrt(float(sz.width * sz.width + sz.height * sz.height)) * fTaoCenters;

		float fThScoreScore = 0.72f;

		// Other constant parameters settings.

		// Gaussian filter parameters, in pre-processing
		Size szPreProcessingGaussKernelSize = Size(5, 5);
		double dPreProcessingGaussSigma = 1.0;

		float fDistanceToEllipseContour = 0.1f; // (Sect. 3.3.1 - Validation)
		float fMinReliability = 0.4;			// Const parameters to discard bad ellipses

		// Initialize Detector with selected parameters
		CEllipseDetectorYaed *yaed = new CEllipseDetectorYaed();
		yaed->SetParameters(szPreProcessingGaussKernelSize,
							dPreProcessingGaussSigma,
							fThPos,
							fMaxCenterDistance,
							iThLength,
							fThObb,
							fDistanceToEllipseContour,
							fThScoreScore,
							fMinReliability,
							iNs);

		// Detect
		vector<Ellipse> ellsYaed;
		Mat1b gray2 = gray.clone();
		yaed->Detect(gray2, ellsYaed);

		/*vector<double> times = yaed.GetTimes();
		cout << "--------------------------------" << endl;
		cout << "Execution Time: " << endl;
		cout << "Edge Detection: \t" << times[0] << endl;
		cout << "Pre processing: \t" << times[1] << endl;
		cout << "Grouping:       \t" << times[2] << endl;
		cout << "Estimation:     \t" << times[3] << endl;
		cout << "Validation:     \t" << times[4] << endl;
		cout << "Clustering:     \t" << times[5] << endl;
		cout << "--------------------------------" << endl;
		cout << "Total:	         \t" << yaed.GetExecTime() << endl;
		cout << "--------------------------------" << endl;*/

		tms.push_back(yaed->GetExecTime());

		vector<Ellipse> gt;
		LoadGT(gt, sWorkingDir + "gt\\" + "gt_" + name_ext + ".txt", false); // Prasad is in radians,set to true

		float pr, re, fm;
		std::tie(pr, re, fm) = Evaluate(gt, ellsYaed, fThScoreScore, image);

		prs.push_back(pr);
		res.push_back(re);
		fms.push_back(fm);

		Mat3b resultImage = image.clone();

		// Draw GT ellipses
		for (unsigned i = 0; i < gt.size(); ++i)
		{
			Ellipse &e = gt[i];
			Scalar color(0, 0, 255);
			ellipse(resultImage, Point(cvRound(e._xc), cvRound(e._yc)), Size(cvRound(e._a), cvRound(e._b)), e._rad * 180.0 / CV_PI, 0.0, 360.0, color, 3);
		}

		yaed->DrawDetectedEllipses(resultImage, ellsYaed);

		//imwrite(out_folder + name + ".png", resultImage);
		//imshow("Yaed", resultImage);
		//waitKey();

		int dbg = 0;
	}

	float N = float(prs.size());
	float sumPR = accumulate(prs.begin(), prs.end(), 0.f);
	float sumRE = accumulate(res.begin(), res.end(), 0.f);
	float sumFM = accumulate(fms.begin(), fms.end(), 0.f);
	double sumTM = accumulate(tms.begin(), tms.end(), 0.0);

	float meanPR = sumPR / N;
	float meanRE = sumRE / N;
	float meanFM = sumFM / N;
	double meanTM = sumTM / N;

	float finalFM = (2.f * meanPR * meanRE) / (meanPR + meanRE);

	cout << "F-measure : " << finalFM << endl;
	cout << "Exec time : " << meanTM << endl;

	getchar();
}

enum Action
{
	annotateImage,
	skeletonImage,
	detectColor,
	detectRectangle
};

int main(int argc, char **argv)
{
	Action action = annotateImage;
	int pathindex = 0;
	if (argc == 2)
	{
		cout << "one argument" << endl;
		pathindex = 1;
	}
	else if (argc == 3)
	{
		cout << "two arguments" << endl;
		pathindex = 2;
		cout << argv[1] << endl;
		if (strcmp(argv[1], "skeleton") == 0)
		{
			action = skeletonImage;
		}
		if (strcmp(argv[1], "color") == 0)
		{
			action = detectColor;
		}
		if (strcmp(argv[1], "rectangle") == 0)
		{
			action = detectRectangle;
		}

	}
	else
	{
		cout << "Expected one or two arguments" << endl;
		return 1;
	}

	char *unresolved_path = argv[pathindex];
	char *resolved_path = (char *)malloc(PATH_MAX);
	realpath(unresolved_path, resolved_path);
	// char *extension = (char *)malloc(20);
	// _splitpath_s(resolved_path, NULL, 0, NULL, 0, NULL, 0, extension, 20);
	// cout << "file extension: " << extension << endl;
	// OnVideo();
	cout << "action: " << action << endl;
	cout << "annotateImage: " << annotateImage << ", skeletonImage: " << skeletonImage << endl;
	switch (action)
	{
	case annotateImage:
		cout << "annotate image" << endl;
		OnImage(resolved_path);
		break;
	case skeletonImage:
		cout << "create skeleton" << endl;
		create_skeleton(resolved_path);
		break;
	case detectColor:
		cout << "detect lines by color" << endl;
		detect_lines_by_color(resolved_path);
		break;
	case detectRectangle:
		cout << "detect rectangles" << endl;
		detect_quadrilateral(resolved_path);
		break;
	default:
		break;
	}
	//OnDataset();
	free(resolved_path);
	// free(extension);
	return 0;
}

// Test on single image
int main2()
{
	string images_folder = "D:\\SO\\img\\";
	string out_folder = "D:\\SO\\img\\";
	vector<string> names;

	glob(images_folder + "Lo3my4.*", names);

	for (const auto &image_name : names)
	{
		string name = image_name.substr(image_name.find_last_of("\\") + 1);
		name = name.substr(0, name.find_last_of("."));

		Mat3b image = imread(image_name);
		Size sz = image.size();

		// Convert to grayscale
		Mat1b gray;
		cvtColor(image, gray, CV_BGR2GRAY);

		// Parameters Settings (Sect. 4.2)
		int iThLength = 16;
		float fThObb = 3.0f;
		float fThPos = 1.0f;
		float fTaoCenters = 0.05f;
		int iNs = 16;
		float fMaxCenterDistance = sqrt(float(sz.width * sz.width + sz.height * sz.height)) * fTaoCenters;

		float fThScoreScore = 0.7f;

		// Other constant parameters settings.

		// Gaussian filter parameters, in pre-processing
		Size szPreProcessingGaussKernelSize = Size(5, 5);
		double dPreProcessingGaussSigma = 1.0;

		float fDistanceToEllipseContour = 0.1f; // (Sect. 3.3.1 - Validation)
		float fMinReliability = 0.5;			// Const parameters to discard bad ellipses

		CEllipseDetectorYaed *yaed = new CEllipseDetectorYaed();
		yaed->SetParameters(szPreProcessingGaussKernelSize,
							dPreProcessingGaussSigma,
							fThPos,
							fMaxCenterDistance,
							iThLength,
							fThObb,
							fDistanceToEllipseContour,
							fThScoreScore,
							fMinReliability,
							iNs);

		// Detect
		vector<Ellipse> ellsYaed;
		Mat1b gray2 = gray.clone();
		yaed->Detect(gray2, ellsYaed);

		vector<double> times = yaed->GetTimes();
		cout << "--------------------------------" << endl;
		cout << "Execution Time: " << endl;
		cout << "Edge Detection: \t" << times[0] << endl;
		cout << "Pre processing: \t" << times[1] << endl;
		cout << "Grouping:       \t" << times[2] << endl;
		cout << "Estimation:     \t" << times[3] << endl;
		cout << "Validation:     \t" << times[4] << endl;
		cout << "Clustering:     \t" << times[5] << endl;
		cout << "--------------------------------" << endl;
		cout << "Total:	         \t" << yaed->GetExecTime() << endl;
		cout << "--------------------------------" << endl;

		Mat3b resultImage = image.clone();
		yaed->DrawDetectedEllipses(resultImage, ellsYaed);

		imwrite(out_folder + name + ".png", resultImage);

		imshow("Yaed", resultImage);
		waitKey();

		int yghds = 0;
	}

	return 0;
}