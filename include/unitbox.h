// Copyright 2016 Dolotov Evgeniy

#ifndef UNITBOX_H
#define UNITBOX_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

#include "neural_network.h"
#include "component.h"
#include "boundingbox_map.h"
#include "confidence_map.h"

const float SAMPLE_INTERSECTION_PERCENT = 0.3;

enum MergeType {
    VERTICAL,
    HORIZONT
};

class UnitboxDetector {
public:
    UnitboxDetector();
    UnitboxDetector(std::string netConfiguration,
                    std::string pretrainNetwork);
    void detect(const cv::Mat& image, std::vector<cv::Rect>& objects);
    ~UnitboxDetector();
private:
    void sampleImage(const cv::Mat& images, int stride,
                     std::vector<cv::Mat>& samples);
    void mergeMaps(int stride, MergeType type,
                   const std::vector<ConfidenceMap>& confidenceMaps,
                   const std::vector<BoundingboxMap>& boundongboxMaps,
                   ConfidenceMap& confidenceMap,
                   BoundingboxMap& boundingboxMap);
    void processSamples(const std::vector<cv::Mat>& samples,
                 std::vector<ConfidenceMap>& confidenceMaps,
                 std::vector<BoundingboxMap>& boundingboxMap);
    void resizeToNetInputSize(const cv::Mat& image, int sideSize,
                              cv::Mat& resizedImage, float& scale);
    NeuralNetwork net;
};

#endif // UNITBOX_H
