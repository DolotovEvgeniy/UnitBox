#include "unitbox.h"

using namespace cv;
using namespace std;

UnitboxDetector::UnitboxDetector() {

}

UnitboxDetector::UnitboxDetector(string netConfiguration,
                                 string pretrainNetwork) {
}

void UnitboxDetector::detect(const Mat& image,
                             vector<Rect>& objects) {
    CV_Assert(image.channels() == 3);

    Size inputSize = net.inputLayerSize();
    Mat resizedImage;
    float scale;
    resizeToNetInputSize(image, inputSize.width, resizedImage, scale);

    int stride = inputSize.width*SAMPLE_INTERSECTION_PERCENT;
    vector<Mat> samples;
    sampleImage(resizedImage, stride, samples);

    vector<ConfidenceMap> confidenceMaps;
    vector<BoundingboxMap> boundingboxMaps;
    processSamples(samples, confidenceMaps, boundingboxMaps);

    ConfidenceMap confidenceMap;
    BoundingboxMap boundingboxMap;
    if(image.cols > image.rows) {
        mergeMaps(stride, HORIZONT,confidenceMaps, boundingboxMaps,
                  confidenceMap, boundingboxMap);
    } else {
        mergeMaps(stride, VERTICAL,confidenceMaps, boundingboxMaps,
                  confidenceMap, boundingboxMap);
    }
    
    vector<Component> components;
    confidenceMap.findComponents(0.5, components);

    vector<Point> centers;
    for(size_t i = 0; i < components.size(); i++) {
        centers.push_back(components[i].center());
    }

    objects.clear();
    for(size_t i = 0; i < components.size(); i++) {
        objects.push_back(boundingboxMap.getRect(centers[i].x, centers[i].y));
    }
}

UnitboxDetector::~UnitboxDetector() {

}

void UnitboxDetector::sampleImage(const Mat& image, int stride,
                                  vector<Mat>& samples) {
    samples.clear();
    if(image.cols > image.rows) {
        while(image.cols > image.rows + stride*samples.size()) {
            Rect roi(stride*samples.size(), 0, image.rows, image.rows);
            Mat sample(image, roi);
            samples.push_back(sample);
        }
        Rect roi(image.cols-image.rows, 0, image.rows, image.rows);
        Mat sample(image, roi);
        samples.push_back(sample);
    } else {
        while(image.rows > image.rows + stride*samples.size()) {
            Rect roi(0, stride*samples.size(), image.cols, image.cols);
            Mat sample(image, roi);
            samples.push_back(sample);
        }
        Rect roi(0, image.rows-image.cols, image.cols, image.cols);
        Mat sample(image, roi);
        samples.push_back(sample);
    }

}

void UnitboxDetector::processSamples(const vector<Mat>& samples,
                                     vector<ConfidenceMap>& confidenceMaps,
                                     vector<BoundingboxMap>& boundingboxMaps) {
    confidenceMaps.clear();
    confidenceMaps.resize(samples.size());

    boundingboxMaps.clear();
    boundingboxMaps.resize(samples.size());

    for(size_t i = 0; i < samples.size(); i++) {
        net.processImage(samples[i], confidenceMaps[i],
                         boundingboxMaps[i]);
    }
}

void UnitboxDetector::mergeMaps(int stride, MergeType type,
                                const vector<ConfidenceMap>& confidenceMaps,
                                const vector<BoundingboxMap>& boundingboxMaps,
                                ConfidenceMap& confidenceMap,
                                BoundingboxMap& boundingboxMap) {
    Size mapSize = confidenceMaps[0].size();
    if(type == VERTICAL) {
        int width = mapSize.width;
        int height = mapSize.height+stride*(confidenceMaps.size()-1);
        ConfidenceMap fullConfMap(width, height);
        BoundingboxMap fullBoxMap(width, height);
        for(int i = 0; i < confidenceMaps.size(); i++) {
            for(int x = 0; x < mapSize.width; x++) {
                for(int y = 0; y < mapSize.height; y++) {
                    uchar oldConfValue = fullConfMap.at(x, y+stride*i);
                    uchar newConfValue = confidenceMaps[i].at(x, y);
                    if(oldConfValue < newConfValue) {
                        fullConfMap.at(x, y+stride*i) = newConfValue;
                    }

                    for(auto c: CHANNELS) {
                        float oldBoxValue = fullBoxMap.at(x, y+stride*i, c);
                        float newBoxValue = boundingboxMaps[i].at(x, y, c);
                        if(oldBoxValue < newBoxValue) {
                            fullBoxMap.at(x, y+stride*i, c) = newBoxValue;
                        }
                    }
                }
            }
        }
    } else if(type == HORIZONT) {
        int width = mapSize.width+stride*(confidenceMaps.size()-1);
        int height = mapSize.height;
        ConfidenceMap fullConfMap(width, height);
        BoundingboxMap fullBoxMap(width, height);
        for(int i = 0; i < confidenceMaps.size(); i++) {
            for(int x = 0; x < mapSize.width; x++) {
                for(int y = 0; y < mapSize.height; y++) {
                    uchar oldConfValue = fullConfMap.at(x+stride*i, y);
                    uchar newConfValue = confidenceMaps[i].at(x, y);
                    if(oldConfValue < newConfValue) {
                        fullConfMap.at(x+stride*i, y) = newConfValue;
                    }
                    for(auto c: CHANNELS) {
                        float oldBoxValue = fullBoxMap.at(x+stride*i, y, c);
                        float newBoxValue = boundingboxMaps[i].at(x, y, c);
                        if(oldBoxValue < newBoxValue) {
                            fullBoxMap.at(x+stride*i, y, c) = newBoxValue;
                        }
                    }
                }
            }
        }
    }
}

void UnitboxDetector::resizeToNetInputSize(const Mat& image, int sideSize,
                                           Mat& resizedImage, float& scale) {

    Size newSize;
    if (image.cols > image.rows) {
        scale = sideSize/(float)image.rows;
        newSize.height = sideSize;
        newSize.width = image.cols*scale;
    } else {
        scale = sideSize/(float)image.cols;
        newSize.width = sideSize;
        newSize.height = image.rows*scale;
    }

    resize(image, resizedImage, newSize);
}
