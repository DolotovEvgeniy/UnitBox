#include "component.h"

using namespace std;
using namespace cv;

Component::Component(vector<Point> points) {
    this->points = points;
}

Point Component::center() {
    Point centerPoint;
    int x = 0, y = 0;
    for(auto point: points) {
        x += point.x;
        y += point.y;
    }

    centerPoint.x = x / static_cast<double>(points.size());
    centerPoint.y = y / static_cast<double>(points.size());

    return centerPoint;
}
