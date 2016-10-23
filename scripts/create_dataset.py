import cv2
from math import atan, tan, cos, sin, fabs
import numpy as np
import h5py
import sys

class Ellipse:
    def __init__(self, major_axis_radius, minor_axis_radius,
                       angle, center_x, center_y) :
        self.major_axis_radius_ = major_axis_radius
        self.minor_axis_radius_ = minor_axis_radius
        self.angle_ = angle
        self.center_x_ = center_x
        self.center_y_ = center_y
    
    def toRect(self) :
        alpha = atan(-self.major_axis_radius_*tan(self.angle_)/self.minor_axis_radius_)
        betta = atan(self.major_axis_radius_/(tan(self.angle_)*self.minor_axis_radius_))
        xMax = self.center_x_+self.minor_axis_radius_*cos(alpha)*cos(self.angle_)
        xMax-=self.major_axis_radius_*sin(alpha)*sin(self.angle_)
        xMin = self.center_x_-self.minor_axis_radius_*cos(alpha)*cos(self.angle_)
        xMin+=self.major_axis_radius_*sin(alpha)*sin(self.angle_);
        yMax = self.center_y_+self.major_axis_radius_*sin(betta)*cos(self.angle_)
        yMax+=self.minor_axis_radius_*cos(betta)*sin(self.angle_);
        yMin = self.center_y_-self.major_axis_radius_*sin(betta)*cos(self.angle_);
        yMin-=self.minor_axis_radius_*cos(betta)*sin(self.angle_);
        xSide = fabs(xMax-xMin);
        ySide = fabs(yMin-yMax);
        return (int(self.center_x_-ySide/2.0),int(self.center_y_-xSide/2.0), int(ySide), int(xSide))
    major_axis_radius_ = 0
    minor_axis_radius_ = 0
    angle_ = 0
    center_x_ = 0
    center_y_ = 0
    
    def draw(self, image, color, thickness) :
        cv2.ellipse(image, (int(self.center_x_), int(self.center_y_)), (int(self.major_axis_radius_), int(self.minor_axis_radius_)), 180*self.angle_/3.14, 0, 360, color, thickness)


def readEllipse(f) :
    line = f.readline()
    p = line.split(" ")
    return Ellipse(float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]))

def inRect(rect, point):
    if rect[0] < point[0] < rect[0]+rect[2] and rect[1] < point[1] < rect[1]+rect[3] :
       return True
    else : 
       return False

def distanceToRect(rect, point) :
    if inRect(rect, point) :
        return (point[0]-rect[0], point[1]-rect[1], rect[0]+rect[2]-point[0], rect[1]+rect[3]-point[1])
    else :
        return (0, 0, 0, 0)
fddb_file = sys.argv[1] 
prefix = sys.argv[2]
f = open(fddb_file, 'r')
fileList = []
index = 1
while True:
    img_path = prefix + f.readline().strip('\n') + ".jpg"
    if prefix+".jpg" == img_path:
        break
    print img_path
    image = cv2.imread(img_path)
    confidence_heatmap = np.zeros((image.shape[0], image.shape[1], 1), dtype = "uint8")
    boundingbox_heatmap = np.zeros((image.shape[0], image.shape[1], 4), dtype = "uint32")
    objectCount = int(f.readline())
    rectList = []
    for i in range(0, objectCount):
        newEllipse = readEllipse(f)
        newEllipse.draw(confidence_heatmap, 255, -1)
        r = newEllipse.toRect()
        rectList.append(r)
        for x in range(0, image.shape[0]):
            for y in  range(0, image.shape[1]):
                for rect in rectList:
                    dist = distanceToRect(rect, (x,y))
                    if dist != (0,0,0,0):
                        boundingbox_heatmap[x,y,0]=dist[0]
                        boundingbox_heatmap[x,y,1]=dist[1]
                        boundingbox_heatmap[x,y,2]=dist[2]
                        boundingbox_heatmap[x,y,3]=dist[3]
    filename = 'train%d.h5' % index
    h = h5py.File(filename, 'w') 
    h['image'] = image
    h['confidence'] = confidence_heatmap
    h['boundingbox']= boundingbox_heatmap
    fileList.append(filename)
    index+=1
    h.close()

with open("list.txt", 'w') as fi:
    for filename in fileList:
        fi.write(filename + '\n')
