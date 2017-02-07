// Copyright 2016 Dolotov Evgeniy

#ifndef BOUNDINGBOX_MAP_H
#define BOUNDINGBOX_MAP_H

#include <array>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

enum BoundingboxMapChannel {
    TOP   = 0,
    DOWN  = 1,
    RIGHT = 2,
    LEFT  = 3
};

const std::vector<BoundingboxMapChannel> CHANNELS = {TOP, DOWN, RIGHT, LEFT};

#define BOUNDINGBOX_MAP_CHANNEL_COUNT 4

class BoundingboxMap {
public:
    BoundingboxMap() = default;
    BoundingboxMap(int width, int height);
    BoundingboxMap(int width, int height, float* data);
    BoundingboxMap(const BoundingboxMap& map);
    BoundingboxMap& operator=(const BoundingboxMap& map);
    cv::Size size() const;
    int channelsCount() const;
    float& at(int x, int y, BoundingboxMapChannel channel);
    float at(int x,int y, BoundingboxMapChannel channel) const;
    cv::Rect getRect(int x, int y) const;
private:
    std::array<cv::Mat, 4> map_;
};

#endif // BOUNDINGBOX_MAP_H
