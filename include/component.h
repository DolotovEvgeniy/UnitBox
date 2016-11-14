// Copyright 2016 Dolotov Evgeniy

#ifndef COMPONENT_H
#define COMPONENT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

class Component {
public:
    Component();
    Component(std::vector<cv::Point> points);
    Component(const Component& component);
    Component& operator=(const Component& component);
    cv::Point center();
private:
    std::vector<cv::Point> points;
};

void findComponent(cv::Mat map, std::vector<Component> components);
#endif // COMPONENT_H
