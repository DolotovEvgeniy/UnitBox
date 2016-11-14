// Copyright 2016 Dolotov Evgeniy

#ifndef CONFIDENCE_MAP_H
#define CONFIDENCE_MAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

#include "component.h"

class ConfidenceMap {
public:
    ConfidenceMap();
    ConfidenceMap(int width, int height);
    ConfidenceMap(int width, int height, float* data, float threshold = 0.5);
    ConfidenceMap(const ConfidenceMap& map);
    ConfidenceMap& operator=(const ConfidenceMap& map);
    cv::Size size() const;
    uchar& at(int x, int y);
    uchar at(int x, int y) const;
    void show(std::string windowName) const;
    void findComponents(std::vector<Component>& components) const;
private:
    cv::Mat map;
};

#endif // CONFIDENCE_MAP_H
