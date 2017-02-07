// Copyright 2016 Dolotov Evgeniy

#ifndef COMPONENT_H
#define COMPONENT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

class Component {
public:
    Component() = default;
    Component(std::vector<cv::Point> points);
    cv::Point center();
private:
    std::vector<cv::Point> points;
};
#endif // COMPONENT_H
