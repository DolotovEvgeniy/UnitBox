#include "boundingbox_map.h"

using namespace std;
using namespace cv;

BoundingboxMap::BoundingboxMap() {

}

BoundingboxMap::BoundingboxMap(int width, int height, float* data) {

}

BoundingboxMap::BoundingboxMap(const BoundingboxMap& map) {

}

BoundingboxMap& BoundingboxMap::operator=(const BoundingboxMap& map) {
   return *this;
}

Rect BoundingboxMap::getRect(int x, int y) const{
    return Rect();
}

float BoundingboxMap::at(int x, int y, int channel) const {
    return 0;
}

float& BoundingboxMap::at(int x, int y, int channel) {
    float i;
    return i;
}

int BoundingboxMap::channels() const {
    return 4;
}

Size BoundingboxMap::size() const {
    return Size(mapDown.cols, mapDown.rows);
}

BoundingboxMap::BoundingboxMap(int width, int height) {
    mapDown = Mat::zeros(height, width, CV_32FC1);
    mapTop =  Mat::zeros(height, width, CV_32FC1);
    mapRight =  Mat::zeros(height, width, CV_32FC1);
    mapLeft =  Mat::zeros(height, width, CV_32FC1);
}
