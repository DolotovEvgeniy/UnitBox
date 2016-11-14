// Copyright 2016 Dolotov Evgeniy

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "unitbox.h"

using namespace cv;
using namespace std;

static const char argsDefs[] =
        "{ | net              |     | Path to configuration file }"
        "{ | weights          |     | Path to pretrained net     }"
        "{ | image            |     | Path to image              }";

void printHelp(std::ostream& os) {
    os << "\tUsage: --net=path/to/configuration.prototxt"
       << " --weights=path/to/pretrained/net.caffemodel"
       << " --image=path/to/image" << std::endl;
}

namespace ReturnCode {
enum {
    Success = 0,
    ConfigFileNotSpecified = 1,
    WeightsFileNotSpecified = 2,
    ImageFileNotSpecified = 3,
    ConfigFileNotFound = 4,
    WeightsFileNotFound = 5,
    ImageFileNotFound = 6
};
};

int parseCommandLine(int argc, char *argv[], Mat& image,
                     string netConfiguration, string pretrainedNet) {
    cv::CommandLineParser parser(argc, argv, argsDefs);

    string imageFileName = parser.get<std::string>("image");
    netConfiguration = parser.get<std::string>("net");
    pretrainedNet = parser.get<std::string>("weights");

    if (netConfiguration.empty() == true) {
        std::cerr << "Net configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }
    if (pretrainedNet.empty() == true) {
        std::cerr << "Pretrained net file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::WeightsFileNotSpecified;
    }
    if (imageFileName.empty() == true) {
        std::cerr << "Image file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ImageFileNotSpecified;
    }

    std::ifstream netFile(netConfiguration);
    if(netFile.is_open() != true) {
        std::cerr << "File '" << netConfiguration
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ConfigFileNotFound;
    } else {
        netFile.close();
    }
    std::ifstream pretrainedNetFile(pretrainedNet);
    if(pretrainedNetFile.is_open() != true) {
        std::cerr << "File '" << pretrainedNet
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::WeightsFileNotFound;
    } else {
        pretrainedNetFile.close();
    }
    image = imread(imageFileName);

    if (!image.data) {
        std::cerr << "File '" << imageFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ImageFileNotFound;
    }

    return ReturnCode::Success;
}

void saveImageWithObjects(string file_name, const Mat& image, const vector<Rect>& objects) {
    Mat imageWithObjects;
    image.copyTo(imageWithObjects);
    for (size_t i = 0; i < objects.size(); i++) {
        rectangle(imageWithObjects, objects[i], Scalar(0, 255, 0));
    }
    imwrite(file_name, imageWithObjects);
}

int main(int argc, char *argv[]) {
    Mat image;
    string netConfiguration, pretrainedNet;
    int returnCode = parseCommandLine(argc, argv, image,
                                      netConfiguration, pretrainedNet);
    if (returnCode != ReturnCode::Success) {
        return returnCode;
    }
    UnitboxDetector detector(netConfiguration, pretrainedNet);
    vector<Rect> detectedObjects;
    detector.detect(image, detectedObjects);
    saveImageWithObjects("result.jpg", image, detectedObjects);

    return ReturnCode::Success;
}
