import random
import cv2
from camera import Camera
import numpy as np
import scipy.spatial.distance
import tornado.gen
from random import randint

from pycontrib.misc.coroutine import reporting_coroutine, unfailable_coroutine
from pycontrib.misc.informer import Informer
        
class DiffExtractor(object):
    
    class BackgroungExtractor(object):
        def __init__(self):
            self._back = None
            self._seq = None
            self._seqN = 0
        def apply(self, frame):
            medianDepth = 40
            if self._back != None:
                if self._back.shape != frame.shape:
                    self._back = None
                    self.apply(frame)
                if self._seq.shape[0] < medianDepth:
                    self._seq = np.append(self._seq, [frame], axis=0)
                else:
                    self._seq[self._seqN] = frame
                self._back = np.uint8(np.median(self._seq, axis=0))
            else:
                self._back = frame.copy()
                self._seq = np.array([self._back])
            self._seqN += 1
            self._seqN %= medianDepth
            return self.get()
        def get(self):
            return self._back
    
    def __init__(self):
        self._backExt = DiffExtractor.BackgroungExtractor()
        self._counter = 0 
        
    def extract(self, frame):
        bluredFrame = cv2.blur(frame, (3,3))
        if 1:#self._counter % 3 == 0:
            back = self._backExt.apply(bluredFrame)
            self._counter = 0
        else:
            back = self._backExt.get()
        self._counter += 1
#         sobel = cv2.cvtColor(np.uint8(np.absolute(cv2.Sobel(cv2.absdiff(back,bluredFrame), cv2.CV_64F, 0, 1, ksize=5))), cv2.COLOR_BGR2GRAY)
#        cv2.imshow('sobel', sobel)
        diff = cv2.absdiff(back,bluredFrame)
#         mask = cv2.inRange(sobel, 100, 255)
#         dilated = cv2.dilate(mask, np.ones((20, 20),np.uint8),iterations = 1)
#         dilated = cv2.erode(dilated, np.ones((5, 50),np.uint8),iterations = 1)
        
#         cv2.imshow('background', np.concatenate((back, bluredFrame, diff), axis=1))
        
        return diff
    
    def getBackImage(self):
        return self._backExt.get()

class OpticalLevel(object):
    
    def __init__(self, **kwargs):
        self.xMode = (kwargs.get('xmode', 'False') == 'True')
        self.piMode = (kwargs.get('pimode', 'True') == 'True')
        self.videoFn = kwargs.get('videofn')
        
        if self.piMode:
            self.camera = Camera(pi_camera=self.piMode)
        else:
            self.camera = Camera(file=self.videoFn)
        self.stream = self.camera.stream()
        
        self.val = 0
        self.motionValCounter = 0
        self.frameCounter = 0
        self.staticVal = None
        self.motionVal = None
        self.motionValPrev = None
        self.val = None
        
        self.diffExtractor = DiffExtractor()
        self.sampleFrame = None
        
        self.rectSize = (250,900)
        self.setRectCrop(0, 0, 0, 60)
        
        self.perspectiveTransformPointsTo = np.array([(self.rectSize[0], self.rectSize[1]), (0, self.rectSize[1]), 
                                                      (self.rectSize[0], 0), (0, 0)], dtype=np.float32)
        
        self.perspectiveTransformPointsFrom = eval(kwargs.get('calibrationpoints', 'None'))
        if self.perspectiveTransformPointsFrom:
            self.perspectiveTransformPointsFrom = np.array(self.perspectiveTransformPointsFrom, dtype=np.float32)
            self.perspectiveTransform = cv2.getPerspectiveTransform(self.perspectiveTransformPointsFrom, self.perspectiveTransformPointsTo)
            self.calibrated = True
        else:
            self.calibrated = False
        
        self.colorCoef = None

        self.colorMapDepth = 30
        self.colorMapEmpty = np.zeros((self.rectCropSize[0], self.colorMapDepth, self.rectCropSize[1], 3), dtype=np.uint8)
        self.colorMapFilled = np.zeros((self.rectCropSize[0], self.colorMapDepth, self.rectCropSize[1], 3), dtype=np.uint8)
        self.colorMapEmptyIndex = np.zeros((self.rectCropSize[0]), dtype=np.int32)
        self.colorMapFilledIndex = np.zeros((self.rectCropSize[0]), dtype=np.int32)
        self.colorMapEmptyMedian = np.zeros((self.rectCropSize[0], self.rectCropSize[1], 3), dtype=np.uint8)
        self.colorMapFilledMedian = np.zeros((self.rectCropSize[0], self.rectCropSize[1], 3), dtype=np.uint8)
        
        self.levelF = open('/tmp/level.val', 'wb', 0)
        self.levelF.write(b'STARTED\n')
        
    def setRectCrop(self, upper, lower, x_shift, width):
        self.rectCrop = (int(upper), int(self.rectSize[1]-lower), int(self.rectSize[0]/2-width/2+x_shift), int(self.rectSize[0]/2+width/2+x_shift))
        self.rectCropSize = (self.rectCrop[1] - self.rectCrop[0], self.rectCrop[3] - self.rectCrop[2])
        
    def imshow(self, imName, image):
        if self.xMode:
            cv2.imshow(imName, image)
        
    def getSampleFrame(self, expertVal = None):
#         frameRes = cv2.resize(self.frame, (int(self.rectified.shape[0]*self.frame.shape[1]/self.frame.shape[0]), self.rectified.shape[0]))
#         for (i, iNext) in ((0,2), (1,0), (2,3), (3,1)):
#             cv2.line(frameRes, (int(self.perspectiveTransformPointsFrom[i][0]*frameRes.shape[1]/self.frame.shape[1]), 
#                                 int(self.perspectiveTransformPointsFrom[i][1]*frameRes.shape[0]/self.frame.shape[0])),
#                                (int(self.perspectiveTransformPointsFrom[iNext][0]*frameRes.shape[1]/self.frame.shape[1]), 
#                                 int(self.perspectiveTransformPointsFrom[iNext][1]*frameRes.shape[0]/self.frame.shape[0])),
#                                (0,255,0))
        self.sampleFrame = self.rectifiedFull.copy()

        toX = self.sampleFrame.shape[1]#-self.sampleFrame.shape[1]
        toY = self.rectCrop[0]
        fromY = self.rectCrop[1]
        for i in range(101):
            y = int(fromY + i*(toY-fromY)/100)
            w=20
            if i%5==0:
                w+=10
            if i%10==0:
                w+=20
                cv2.putText(self.sampleFrame, '{0}'.format(i),(toX-w-10,y-2), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),0,cv2.LINE_AA)
            cv2.line(self.sampleFrame, (toX-w, y), (toX, y), (255,255,0))
#         if expertVal:
#             cv2.line(self.sampleFrame, (toX-self.rectSize[0], int(self.toY-expertVal*(self.toY-self.fromY)/100)), (toX, int(self.toY-expertVal*(self.toY-self.fromY)/100)), (0, 255, 255))
#             cv2.putText(self.sampleFrame, 'dV = {0:2.1f} %'.format(self.val-expertVal),(280, 880), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
        for line in (((self.rectCrop[2], self.rectCrop[0]),(self.rectCrop[3], self.rectCrop[0])),
                     ((self.rectCrop[3], self.rectCrop[0]),(self.rectCrop[3], self.rectCrop[1])),
                     ((self.rectCrop[3], self.rectCrop[1]),(self.rectCrop[2], self.rectCrop[1])),
                     ((self.rectCrop[2], self.rectCrop[1]),(self.rectCrop[2], self.rectCrop[0]))):
            cv2.line(self.sampleFrame, line[0], line[1], (0, 255, 255))
        cv2.line(self.sampleFrame, (toX-self.rectSize[0], int(self.val)+self.rectCrop[0]), (toX, int(self.val)+self.rectCrop[0]), (255, 255, 0))
#         cv2.putText(self.sampleFrame, 'V={0:2.1f} %'.format(self.percent),(10, self.sampleFrame.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
#         cv2.putText(self.sampleFrame, 'CCF={0}'.format(self.colorCoef),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        return self.sampleFrame #cv2.resize(self.sampleFrame, (int(self.sampleFrame.shape[1]/2), int(self.sampleFrame.shape[0]/2)))
    
    def calibrate(self):
        cFrame = cv2.blur(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY), (3,3))
        cFrame = cv2.Canny(cFrame,50,180)
        image, contours, h = cv2.findContours(cFrame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        acontours = []
        for contour in contours:
            acontour = cv2.approxPolyDP(contour, 7, True)
            if acontour.shape[0] == 4:
                x1, y1, x2, y2, x3, y3, x4, y4 = acontour[0][0][0], acontour[0][0][1], acontour[1][0][0], acontour[1][0][1], acontour[2][0][0], acontour[2][0][1], acontour[3][0][0], acontour[3][0][1]
                acontours.append((acontour, ((x1+x2+x3+x4)/4, (y1+y2+y3+y4)/4), ( max((x1,x2,x3,x4))-min((x1,x2,x3,x4)), max((y1,y2,y3,y4))-min((y1,y2,y3,y4)) )))
                col = random.randint(0,255)
                for p in acontour:
                    cv2.circle(cFrame, (p[0][0], p[0][1]), 10, col)
                    cv2.circle(self.frame, (p[0][0], p[0][1]), 10, (0,255,0))
        if len(acontours) < 8:
            return None
        points = []
        for (extAContour, extCentroid, extSize) in acontours:
            if extSize[0] < 30 or extSize[1] < 30:
                continue
            for (intAContour, intCentroid, intSize) in acontours:
                if not (extSize[0]*0.4 < intSize[0] < extSize[0]*0.6 and extSize[1]*0.4 < intSize[1] < extSize[1]*0.6):
                    continue
                if abs(extCentroid[0]-intCentroid[0]) > 10 or abs(extCentroid[1]-intCentroid[1]) > 10:
                    continue
                points.append((int((extCentroid[0]+intCentroid[0])/2),int((extCentroid[1]+intCentroid[1])/2)))
                cv2.circle(cFrame, (points[-1][0], points[-1][1]), 20, 255)
                cv2.circle(self.frame, (points[-1][0], points[-1][1]), 20, (0,255,0))
                break
            
        #filter doubles
        doubleThreshold = 20
        basicPoints = []
        for p in sorted(points, key=lambda p: p[0]**2+p[1]**2):
            if len([b for b in basicPoints if abs(b[0]-p[0]) < doubleThreshold and abs(b[1]-p[1]) < doubleThreshold]):
                continue
            else:
                basicPoints.append(p)
                
        #self.imshow('cframe', cFrame)
        print(basicPoints)
        if len(basicPoints) != 4:
            return None
        self.perspectiveTransformPointsFrom = np.array(basicPoints, dtype=np.float32)
        self.perspectiveTransform = cv2.getPerspectiveTransform(self.perspectiveTransformPointsFrom, self.perspectiveTransformPointsTo)
        self.calibrated = True
        return True
                   
    def colorCalibration(self):
        return (1, 1, 1)
        elementSize = int(self.rectifiedFull.shape[1]*0.09)
        colors = np.zeros((elementSize*8, 3))
        for i in range(elementSize):
            for e in range(8):
                if e == 0:
                    (y, x) = (elementSize, i)
                elif e == 1:
                    (y, x) = (i, elementSize - 1)
                elif e == 2:
                    (y, x) = (elementSize - 1, self.rectifiedFull.shape[1] - 1 - i)
                elif e == 3:
                    (y, x) = (i, self.rectifiedFull.shape[1] - elementSize)
                elif e == 4:
                    (y, x) = (self.rectifiedFull.shape[0] - elementSize, i)
                elif e == 5:
                    (y, x) = (self.rectifiedFull.shape[0] - 1 - i, elementSize - 1)
                elif e == 6:
                    (y, x) = (self.rectifiedFull.shape[0] - elementSize, self.rectifiedFull.shape[1] - 1 - i)
                else:
                    (y, x) = (self.rectifiedFull.shape[0] - 1 - i, self.rectifiedFull.shape[1] - elementSize)
                colors[i + e*elementSize] = self.rectifiedFull[y,x]
#                 cv2.circle(self.rectifiedFull, (x,y), 1, (0,255,0))

        white = np.median(colors, axis=0)
        if 0 in white:
            return (1, 1, 1)
        self.colorCoef = (255, 255, 255) / white
        return self.colorCoef
    
#     def motionLevel(self, frame):
            
#     @profile
    def meassure(self, frame):
        if self.val == None:
            self.val = 0
        
        self.frameCounter += 1
        self.frame = frame
        if not self.calibrated: # or self.frameCounter % 200 == 10:
            self.calibrate()

        if not self.calibrated:
            return self.val
        
        self.rectifiedFull = cv2.warpPerspective(frame, self.perspectiveTransform, self.rectSize)
#         self.rectifiedFull *= self.colorCalibration()
        self.imshow('rect', self.rectifiedFull)
        self.rectified = self.rectifiedFull[self.rectCrop[0]:self.rectCrop[1], self.rectCrop[2]:self.rectCrop[3]]
#         self.rectifiedMedian = np.uint8(np.median(self.rectified, axis=1))
#         self.rectifiedMedian = np.reshape(self.rectifiedMedian, (self.rectifiedMedian.shape[0], 1, 3))
        
        b = cv2.GaussianBlur(self.rectified, (3, 3), 0)
        g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        diff = self.diffExtractor.extract(g)
        m = np.amax(diff)
        diff = cv2.inRange(diff, int(m)*0.7, 255)
        self.imshow('g', diff)
#         diff = np.reshape(diff, (diff.shape[0], 1))
                
        #check for huge image changes
#         backImage = self.diffExtractor.getBackImage()
#         backImage = np.uint8(np.median(backImage, axis=1))
#         backImage = np.reshape(backImage, (backImage.shape[0], 1, 3))
        changes = np.average(np.reshape(diff, (diff.size, 1))) #scipy.spatial.distance.euclidean(backImage.reshape(backImage.shape[0]*backImage.shape[1]*3), 
        if changes > 0.5:
            return self.val
        
        m = cv2.moments(diff, 1)
        if m['m00']:
            motionVal = m['m01']/m['m00']
            self.motionVal = motionVal
            self.motionValPrev = motionVal
            self.motionValCounter = 0
        else:
            self.motionVal = None
            self.motionValCounter += 1
            l = cv2.cvtColor(np.uint8(np.absolute(cv2.Sobel(self.rectified, cv2.CV_64F, 2, 0, ksize=3))*0.4), cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(l, 20, 255)
            dilated = cv2.dilate(mask, np.ones((4, 10),np.uint8),iterations = 1)
            dilated = cv2.erode(dilated, np.ones((60, 30),np.uint8),iterations = 1)
            dilated = cv2.dilate(dilated, np.ones((15, self.rectified.shape[1]/2),np.uint8),iterations = 4)
            self.imshow('1', dilated)
            indexes = np.argwhere(dilated[:,dilated.shape[1]/2])
            if indexes.any():
                staticVal2 = np.argwhere(dilated[:,dilated.shape[1]/2])[-1][0]
                self.staticVal = staticVal2
            else:
                self.staticVal = None
        
        val = None
        if self.motionVal != None:
            val = self.motionVal
        elif self.motionValCounter > 20:
            val = self.staticVal
        if val != None:
            self.val = self.val*0.90 + val*0.10
            self.levelF.write('{0}\n'.format(self.percent).encode())

        return self.val
            
    @property
    def percent(self):
        return (self.rectCrop[1]-self.val)*100 / self.rectCropSize[0]
    
    @property
    def state(self):
        if not self.calibrated:
            return 'Calibrating'
        if self.val == None:
            return 'Learning'
        return 'Levelling'
        
    def sourcePointToVal(self, point):
        return self.rectYToVal(cv2.perspectiveTransform(np.array([(point,)],dtype=np.float32), self.perspectiveTransform)[0][0][1])
        
    @unfailable_coroutine
    @tornado.gen.coroutine
    def start(self):
        cnt = 0
        while 1:
            ret, frame = next(self.stream)
            if not ret:
                return
            cnt += 1
            cnt %= 3
            if cnt:
                continue
            self.meassure(frame)

            if self.xMode:
                self.imshow('sampleFrame', self.getSampleFrame())
                cv2.waitKey(100)
            yield tornado.gen.sleep(0.01)
        
