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
    
    def scale(self, scaleCoeff) :
        self.major_axis_radius_ *= scaleCoeff
        self.minor_axis_radius_ *= scaleCoeff
        self.center_x_ *= scaleCoeff
        self.center_y_ *= scaleCoeff

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
    
    def draw(self, image, color, thickness) :
        cv2.ellipse(image, (int(self.center_x_), int(self.center_y_)), (int(self.major_axis_radius_), int(self.minor_axis_radius_)), 180*self.angle_/3.14, 0, 360, color, thickness)
    
    major_axis_radius_ = 0
    minor_axis_radius_ = 0
    angle_ = 0
    center_x_ = 0
    center_y_ = 0


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
data_directory = sys.argv[3]
sampleSize = int(sys.argv[4])
f = open(fddb_file, 'r')
fileList = []
index = 1
while True:
    img_path = prefix + f.readline().strip('\n') + ".jpg"
    if prefix+".jpg" == img_path:
        break
    print img_path
    image = cv2.imread(img_path)
    cv2.imwrite("img1.png", image)
    scale = 0
    oldSize = (image.shape[1], image.shape[0])
    newSize = 0
    if image.shape[0] < image.shape[1] :
        scale = float(sampleSize)/image.shape[0]
        newSize = (int(image.shape[1]*scale), sampleSize)
    else :
        scale = float(sampleSize)/image.shape[1]
        newSize = (sampleSize, int(image.shape[0]*scale))
    print "resize: ", oldSize, "->", newSize
    print scale
    image = cv2.resize(image, newSize)
    cv2.imwrite("img.png", image)
    images = np.zeros((1, 3, image.shape[0], image.shape[1]), dtype = "uint8")
    confidence_heatmaps = np.zeros((1, 1, image.shape[0], image.shape[1]), dtype = "uint8")
    
    boundingbox_heatmaps = np.zeros((1, 4,image.shape[0], image.shape[1]), dtype = "uint32")

    confidence_heatmap = np.zeros((image.shape[0], image.shape[1], 1), dtype = "uint8")
    boundingbox_heatmap = np.zeros((image.shape[0], image.shape[1], 4), dtype = "uint32")

    objectCount = int(f.readline())
    rectList = []
    for i in range(0, objectCount):
        newEllipse = readEllipse(f)
        newEllipse.scale(scale)
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
    cv2.imshow("conf", confidence_heatmap)
    cv2.waitKey(0)
    confidence_heatmaps[0] = np.transpose(confidence_heatmap, (2, 0, 1))
    images[0] = np.transpose(image, (2, 0, 1))
    boundingbox_heatmaps[0] = np.transpose(boundingbox_heatmap, (2, 0, 1))
    filename = sys.argv[3]+'train%d.h5' % index
    h = h5py.File(filename, 'w') 
    h['image'] = images
    h['confidence'] = confidence_heatmaps
    h['boundingbox']= boundingbox_heatmaps
    fileList.append(filename)
    index+=1
    h.close()

with open("list.txt", 'w') as fi:
    for filename in fileList:
        fi.write(filename + '\n')
