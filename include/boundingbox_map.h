// Copyright 2016 Dolotov Evgeniy

#ifndef BOUNDINGBOX_MAP_H
#define BOUNDINGBOX_MAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class BoundingboxMap {
public:
    BoundingboxMap();
    BoundingboxMap(int width, int height);
    BoundingboxMap(int width, int height, float* data);
    BoundingboxMap(const BoundingboxMap& map);
    BoundingboxMap& operator=(const BoundingboxMap& map);
    cv::Size size() const;
    int channels() const;
    float& at(int x, int y, int channel);
    float at(int x,int y, int channel) const;
    cv::Rect getRect(int x, int y) const;
private:
    cv::Mat mapTop, mapDown, mapRight, mapLeft;
};

#endif // BOUNDINGBOX_MAP_H
