#include "confidence_map.h"
#include <iostream>

using namespace std;
using namespace cv;

ConfidenceMap::ConfidenceMap() {

}

ConfidenceMap::ConfidenceMap(int width, int height, float* data, float threshold) {
    Mat floatMap(height, width, CV_32FC1, data);
    cv::threshold(floatMap, map, threshold);
}

ConfidenceMap::ConfidenceMap(const ConfidenceMap& map) {
    map.map.copyTo(this->map);
}

ConfidenceMap& ConfidenceMap::operator=(const ConfidenceMap& map) {
    map.map.copyTo(this->map);
    return *this;
}

uchar& ConfidenceMap::at(int x, int y) {
    return map.at<float>(y, x);
}

uchar ConfidenceMap::at(int x, int y) const {
    return map.at<float>(y, x);
}

void ConfidenceMap::show(std::string windowName) const {
    imshow("Confidence map", map);
}

void ConfidenceMap::findComponents(std::vector<Component>& components) const {
    cout << "TODO: findComponents" << endl;
}

Size ConfidenceMap::size() const {
    return Size(map.cols, map.rows);
}

ConfidenceMap::ConfidenceMap(int width, int height) {
    map = Mat::zeros(height, width, CV_8UC1);
}
