#include "boundingbox_map.h"
#include <cassert>
using namespace std;
using namespace cv;

BoundingboxMap::BoundingboxMap(int width, int height, const float* data) {
    float *tmpData = new float[width*height*channelsCount()];
    for(int i = 0; i<width*height*channelsCount(); i++) {
        tmpData[i] = data[i];
    }
    for(int i = 0; i < channelsCount(); i++) {
        map_[i] = Mat(height, width, CV_32FC1, tmpData + width*height*i);
         cv::normalize(map_[i], map_[i], 0, 1, cv::NORM_MINMAX);
         imshow("fff", map_[i]);
         waitKey(0);
    }
}

BoundingboxMap::BoundingboxMap(const BoundingboxMap& map) {
    for(auto channel: CHANNELS) {
        this->map_.at(channel) = map.map_.at(channel);
    }
}

BoundingboxMap& BoundingboxMap::operator=(const BoundingboxMap& map) {
    for(auto channel: CHANNELS) {
        this->map_.at(channel) = map.map_.at(channel);
    }
    return *this;
}

Rect BoundingboxMap::getRect(int x, int y) const{
    Point leftTopPoint;
    leftTopPoint.x = x - map_.at(LEFT).at<float>(Point(x, y));
    leftTopPoint.y = y - map_.at(TOP).at<float>(Point(x, y));

    int width  = map_.at(LEFT).at<float>(Point(x, y))
                 + map_.at(RIGHT).at<float>(Point(x, y));

    int height = map_.at(TOP).at<float>(Point(x, y))
                 + map_.at(DOWN).at<float>(Point(x, y));
    return Rect(leftTopPoint, Size(width, height));
}

float BoundingboxMap::at(int x, int y, BoundingboxMapChannel channel) const {
    assert(x >= 0 && y >= 0 && x < size().width && y < size().height);

    return map_.at(channel).at<float>(Point(x, y));
}

float& BoundingboxMap::at(int x, int y, BoundingboxMapChannel channel) {
    assert(x >= 0 && y >= 0 && x < size().width && y < size().height);

    return map_.at(channel).at<float>(Point(x, y));
}

int BoundingboxMap::channelsCount() const {
    return map_.size();
}

Size BoundingboxMap::size() const {
    return Size(map_.at(0).cols, map_.at(0).rows);
}

BoundingboxMap::BoundingboxMap(int width, int height) {
    for(auto& channel: map_) {
        channel = Mat::zeros(height, width, CV_32FC1);
    }
}
