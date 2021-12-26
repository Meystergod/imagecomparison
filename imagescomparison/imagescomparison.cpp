#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/core/utils/logger.hpp"

#define SUCCESS 1
#define END_OF_PROGRAM 0
#define OPEN_IMAGE_ERROR -1

using namespace std;
using namespace cv;

int image_surf_process(Mat current_image, Mat next_image);
int image_comparison(vector<string> paths, int accuracy);
void input_handler(short int& accuracy, vector<string>& paths);
bool path_checker(const string& path);
bool is_uint(const string& value);

int image_surf_process(Mat current_image, Mat next_image)
{
	BFMatcher matcher;
	Mat current_image_desc, next_image_desc;
	vector<vector<DMatch>> matches;
	vector<DMatch> best_matches;
	vector<KeyPoint> current_image_kp, next_image_kp;

	Ptr<xfeatures2d::SURF> surf_detector = xfeatures2d::SURF::create();
	surf_detector->detectAndCompute(current_image, noArray(), current_image_kp, current_image_desc);
	surf_detector->detectAndCompute(next_image, noArray(), next_image_kp, next_image_desc);

	matcher.knnMatch(current_image_desc, next_image_desc, matches, 5);

	for_each(matches.begin(), matches.end(), [&](vector<DMatch> match)
	{
		if (match.size() > 1 && match[0].distance / match[1].distance <= 0.75)
		{
			best_matches.push_back(match[0]);
		}
	});

	int number_kp = (current_image_kp.size() <= next_image_kp.size()) ? current_image_kp.size() : current_image_kp.size();
	int compare_percent = (double)best_matches.size() / (double)number_kp * (double)100;

	return compare_percent;
}

int image_comparison(vector<string> paths, int accuracy)
{
	Mat current_image, next_image;
	int compare_percent;

	cout << endl << "Comparison results:" << endl;
	for (int i = 0; i < paths.size(); i++)
	{
		current_image = imread(paths[i], IMREAD_GRAYSCALE);
		if (current_image.empty())
		{
			cout << "Failed: could not open the image: " << paths[i] << endl;
			return OPEN_IMAGE_ERROR;
		}
		for (int j = i + 1; j < paths.size(); j++)
		{
			next_image = imread(paths[j], IMREAD_GRAYSCALE);
			if (next_image.empty())
			{
				cout << "Failed: could not open the image: " << paths[j] << endl;
				return OPEN_IMAGE_ERROR;
			}
			compare_percent = image_surf_process(current_image, next_image);
			if (compare_percent >= accuracy)
			{
				cout << paths[i] << ", " << paths[j] << ", " << compare_percent << endl;
			}
		}
	}

	return SUCCESS;
}

void input_handler(short int& accuracy, vector<string>& paths)
{
	string path;
	string value;

	for (;;)
	{
		cout << "Accuracy: ";
		cin >> value;
		if (!is_uint(value) || atoi(value.c_str()) < 0 || atoi(value.c_str()) > 100)
		{
			cout << "Failed: incorrect value of accuracy." << endl;
			continue;
		}
		accuracy = atoi(value.c_str());
		break;
	}
	cout << endl;
	for (;;)
	{
		cout << "Path to image (enter the ~ to finish input of paths): ";
		cin >> path;
		if (path == "~" && paths.size() >= 2)
		{
			break;
		}
		if (path == "~" && paths.size() < 2)
		{
			cout << "Failed: total number of input images is less then 2." << endl;
			continue;
		}
		if (!path_checker(path))
		{
			cout << "Failed: incorrect path to the image." << endl;
			continue;
		}
		paths.push_back(path);
	}
}

bool path_checker(const string& path)
{
	struct stat buffer;
	return (stat(path.c_str(), &buffer) == 0);
}

bool is_uint(const string& value)
{
	return value.find_first_not_of("0123456789") == string::npos;
}

int main(int argc, char* argv[])
{
	utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

	vector<string> paths;
	short int accuracy, response;

	input_handler(accuracy, paths);
	response = image_comparison(paths, accuracy);
	if (response != SUCCESS)
	{
		system("pause");
		return response;
	}

	system("pause");
	return END_OF_PROGRAM;
}