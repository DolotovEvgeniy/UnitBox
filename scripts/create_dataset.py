import cv2
from math import atan, tan, cos, sin, fabs
import numpy as np
import h5py
import sys
import random

class Ellipse :
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

def inRect(rect, point) :
    if rect[0] < point[0] < rect[0]+rect[2] and rect[1] < point[1] < rect[1]+rect[3] :
       return True
    else : 
       return False

def distanceToRect(rect, point) :
    if inRect(rect, point) :
        return (point[0]-rect[0], point[1]-rect[1], rect[0]+rect[2]-point[0], rect[1]+rect[3]-point[1])
    else :
        return (0, 0, 0, 0)


fddbFilePath = sys.argv[1] 
prefix = sys.argv[2]
dataDirectory = sys.argv[3]
sampleSize = int(sys.argv[4])
intersectionPercent = float(sys.argv[5])
stride = int(sampleSize * (1 - intersectionPercent))
setProbability = float(sys.argv[6])
fddbFile = open(fddbFilePath, 'r')

trainList = []
testList = []
testIndex = 1
trainIndex = 1
while True:
    imageName = fddbFile.readline().strip('\n')
    if imageName == "":
        break

    imagePath = prefix + imageName + ".jpg"
    print imagePath
    image = cv2.imread(imagePath)

    imageSize = (image.shape[1], image.shape[0])
    scale = 0
    resizedImageSize = 0
    orientation = 0 # 0 - vertical, 1 - horizontal
    if image.shape[0] < image.shape[1] : 
        scale = float(sampleSize)/image.shape[0]
        resizedImageSize = (int(image.shape[1]*scale), sampleSize)
        orientation = 1
    else :
        scale = float(sampleSize)/image.shape[1]
        resizedImageSize = (sampleSize, int(image.shape[0]*scale))
        orientation = 0

    resizedImage = cv2.resize(image, resizedImageSize)

    confidenceMap = np.zeros((resizedImage.shape[0], resizedImage.shape[1], 1), dtype = "uint8")
    boundingboxMap = np.zeros((resizedImage.shape[0], resizedImage.shape[1], 4), dtype = "uint16")

    objectCount = int(fddbFile.readline())
    rectList = []
    for i in range(0, objectCount):
        newEllipse = readEllipse(fddbFile)
        newEllipse.scale(scale)
        newEllipse.draw(confidenceMap, 255, -1)
        r = newEllipse.toRect()
        rectList.append(r)
        for x in range(0, resizedImage.shape[0]):
            for y in  range(0, resizedImage.shape[1]):
                for rect in rectList:
                    dist = distanceToRect(rect, (x,y))
                    if dist != (0,0,0,0):
                        boundingboxMap[x,y,0]=dist[0]
                        boundingboxMap[x,y,1]=dist[1]
                        boundingboxMap[x,y,2]=dist[2]
                        boundingboxMap[x,y,3]=dist[3]

    #image width - image.shape[1], image height - image.shape[0]
    border = 0
    if orientation == 0 :
        border = resizedImage.shape[0]
    if orientation == 1 :
        border = resizedImage.shape[1]

    sampleCount = 0
    while stride * sampleCount + sampleSize < border :
        if orientation == 0 : # vertical image height > width
            x = 0
            y =  stride * sampleCount
        if orientation == 1 : # horizontal image width > height
            x = stride * sampleCount
            y = 0
        cutImage = resizedImage[y:y+sampleSize, x:x+sampleSize]
        cutConfidenceMap = confidenceMap[y:y+sampleSize, x:x+sampleSize]
        cutBoundingboxMap = boundingboxMap[y:y+sampleSize, x:x+sampleSize]

        images = np.zeros((1, 3, sampleSize, sampleSize), dtype = "uint8")
        confidenceMaps = np.zeros((1, 1, sampleSize, sampleSize), dtype = "uint8")
        boundingboxMaps = np.zeros((1, 4, sampleSize, sampleSize), dtype = "uint16")
        confidenceMaps[0] = np.transpose(cutConfidenceMap, (2, 0, 1))
        images[0] = np.transpose(cutImage, (2, 0, 1))
        boundingboxMaps[0] = np.transpose(cutBoundingboxMap, (2, 0, 1))
        sampleSetProbability = random.random()
        filename = ""
        if sampleSetProbability > setProbability :
            filename = sys.argv[3]+'train%d.h5' % trainIndex
            trainList.append(filename)
            trainIndex += 1
        else :
            filename = sys.argv[3]+'test%d.h5' % testIndex
            testList.append(filename)
            testIndex += 1
        h = h5py.File(filename, 'w') 
        h['image'] = images
        h['confidence'] = confidenceMaps
        h['boundingbox'] = boundingboxMaps
        h.close()
        sampleCount += 1 

print "-----Result-----"
print "Train sample count:", trainIndex - 1
print "Test sample count:", testIndex - 1
with open("train.txt", 'w') as fi :
    for filename in trainList :
        fi.write(filename + '\n')
with open("test.txt", 'w') as fi :
    for filename in testList :
        fi.write(filename + '\n')
