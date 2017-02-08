#include "confidence_map.h"
#include <iostream>

using namespace std;
using namespace cv;

ConfidenceMap::ConfidenceMap(int width, int height, const float* data) {
    float *tmpData = new float[width*height];
    for(int i = 0; i<width*height; i++) {
        tmpData[i] = data[i];
    }
    map = Mat(height, width, CV_32FC1, tmpData);
}

ConfidenceMap::ConfidenceMap(const ConfidenceMap& map) {
    map.map.copyTo(this->map);
}

ConfidenceMap& ConfidenceMap::operator=(const ConfidenceMap& map) {
    map.map.copyTo(this->map);
    return *this;
}

float& ConfidenceMap::at(int x, int y) {
    return map.at<float>(Point(x, y));
}

float ConfidenceMap::at(int x, int y) const {
    return map.at<float>(Point(x, y));
}

void ConfidenceMap::show(std::string windowName) const {
    Mat picture;
    cv::normalize(map, picture, 0, 1, cv::NORM_MINMAX);
    threshold(picture, picture, 0.5, 255, THRESH_BINARY);
    picture.convertTo(picture, CV_8UC1);
    imshow(windowName, picture);
    imwrite("op.jpg", picture);
}

void ConfidenceMap::findComponents(float thresholdValue,
                                   std::vector<Component>& components) const {
    Mat binaryMask;
    threshold(map, binaryMask, thresholdValue, 1, THRESH_BINARY);
    cout << "TODO: findComponents" << endl;
}

Size ConfidenceMap::size() const {
    return Size(map.cols, map.rows);
}

ConfidenceMap::ConfidenceMap(int width, int height) {
    map = Mat::zeros(height, width, CV_8UC1);
}
