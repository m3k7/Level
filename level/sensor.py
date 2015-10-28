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
                assert(self._back.shape == frame.shape)
#                 self._seq = np.append(self._seq, [frame], axis=0)
                if self._seq.shape[0] < medianDepth:
                    self._seq = np.append(self._seq, [frame], axis=0)
                else:
                    self._seq[self._seqN] = frame
#                     np.delete(self._seq, 0, 0)
                self._back = np.uint8(np.median(self._seq, axis=0))
#                 self._back = np.uint8((np.float16(self._back)*9+np.float16(frame)) / 10)
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
        sobel = cv2.cvtColor(np.uint8(np.absolute(cv2.Sobel(cv2.absdiff(back,bluredFrame), cv2.CV_64F, 0, 1, ksize=5))), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(back,bluredFrame)
        mask = cv2.inRange(sobel, 190, 255)
        dilated = cv2.dilate(mask, np.ones((20, 20),np.uint8),iterations = 1)
        dilated = cv2.erode(dilated, np.ones((5, 50),np.uint8),iterations = 1)
        
#         cv2.imshow('background', np.concatenate((back, bluredFrame, diff), axis=1))
#         cv2.imshow('mask', np.concatenate((sobel, mask, dilated), axis=1))
        
        return dilated
    
    def getBackImage(self):
        return self._backExt.get()

class OpticalLevel(object):
    
    def __init__(self, **kwargs):
        self.xMode = (kwargs.get('xmode', 'False') == 'True')
        self.piMode = (kwargs.get('pimode', 'True') == 'True')
        self.videoFn = kwargs.get('videofn')
        self.upperBound = float(kwargs.get('upperbound', '0'))
        self.lowerbound = float(kwargs.get('lowerbound', '1'))
        
        if self.piMode:
            self.camera = Camera(pi_camera=self.piMode)
        else:
            self.camera = Camera(file=self.videoFn)
        self.stream = self.camera.stream()
        
        self.val = 0
        self.frameCounter = 0
        self.staticVal = None
        self.motionVal = None
        self.motionValPrev = None
        self.val = None
        
        self.diffExtractor = DiffExtractor()
        self.sampleFrame = None
        
        self.rectSize = (200,800)
        self.correctedRectSize = (200, 600)
        self.rectCrop = (int(self.rectSize[1]*0), int(self.rectSize[1]), int(self.rectSize[0]*0.4), int(self.rectSize[0]*0.6))
        self.rectCropSize = (self.rectCrop[1] - self.rectCrop[0], self.rectCrop[3] - self.rectCrop[2])
        
        self.perspectiveTransformPointsTo = np.array([(self.rectSize[0], self.rectSize[1]*self.lowerbound), (0, self.rectSize[1]*self.lowerbound), 
                                                      (self.rectSize[0], self.rectSize[1]*self.upperBound), (0, self.rectSize[1]*self.upperBound)], dtype=np.float32)
        
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
        toY = self.sampleFrame.shape[0]
        for i in range(101):
            y = int(toY - i*(toY)/100)
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
        cv2.line(self.sampleFrame, (toX-self.rectSize[0], int(self.val)), (toX, int(self.val)), (255, 255, 0))
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
                
        self.imshow('cframe', cFrame)
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
            
#     @profile
    def meassure(self, frame):
        if self.val == None:
            self.val = 0
        
        self.frameCounter += 1
        self.frame = frame
        if not self.calibrated or self.frameCounter % 200 == 0:
            self.calibrate()

        if not self.calibrated:
            return self.val
        
        self.rectifiedFull = cv2.warpPerspective(frame, self.perspectiveTransform, self.rectSize)
        self.rectifiedFull *= self.colorCalibration()
        self.imshow('rect', self.rectifiedFull)
        self.rectified = self.rectifiedFull[self.rectCrop[0]:self.rectCrop[1], self.rectCrop[2]:self.rectCrop[3]]
        self.rectifiedMedian = np.uint8(np.median(self.rectified, axis=1))
        self.rectifiedMedian = np.reshape(self.rectifiedMedian, (self.rectifiedMedian.shape[0], 1, 3))
        
        diff = self.diffExtractor.extract(cv2.blur(self.rectified, (10,1)))
        diff = np.uint8(np.median(diff, axis=1))
#         diff = np.reshape(diff, (diff.shape[0], 1))
                
        #check for huge image changes
        backImage = self.diffExtractor.getBackImage()
        backImage = np.uint8(np.median(backImage, axis=1))
        backImage = np.reshape(backImage, (backImage.shape[0], 1, 3))
        changes = scipy.spatial.distance.euclidean(backImage.reshape(backImage.shape[0]*backImage.shape[1]*3), self.rectifiedMedian.reshape(self.rectifiedMedian.shape[0]*self.rectifiedMedian.shape[1]*3))
        print(changes)
#         changes = 10
        if changes > 400:
            return self.val
        
        m = cv2.moments(diff, 1)
        if m['m00']:
            motionVal = m['m01']/m['m00']
#             if self.motionValPrev==None: # or abs(self.motionValPrev - motionVal) < diff.shape[0]/5: 
            self.motionVal = motionVal
            self.motionValPrev = motionVal
        else:
            self.motionVal = None

        if not self.motionVal:
            if randint(0,20) == 0:
                #delearn color detector
                for y in range(self.rectified.shape[0]):
                    self.colorMapEmpty[y, self.colorMapEmptyIndex[y]] = np.zeros((self.rectified.shape[1],3), dtype=np.ubyte)
                    self.colorMapEmptyIndex[y] += 1
                    self.colorMapEmptyIndex[y] %= self.colorMapDepth
                    self.colorMapFilled[y, self.colorMapFilledIndex[y]] = np.zeros((self.rectified.shape[1],3), dtype=np.ubyte)
                    self.colorMapFilledIndex[y] += 1
                    self.colorMapFilledIndex[y] %= self.colorMapDepth
                self.colorMapEmptyMedian = np.uint8(np.median(self.colorMapEmpty, axis=1))
                self.colorMapFilledMedian = np.uint8(np.median(self.colorMapFilled, axis=1))
        else:
            #learn color detector            
            for y in range(self.rectified.shape[0]):
                if abs(y-self.motionVal)<5:
                    continue
                if (y<self.motionVal):
                    colorMap = self.colorMapEmpty
                    colorMapIndex = self.colorMapEmptyIndex
                else:
                    colorMap = self.colorMapFilled
                    colorMapIndex = self.colorMapFilledIndex
                colorMap[y, colorMapIndex[y]] = self.rectified[y] # self.rectified[y, int(self.rectified.shape[1]/2)] #(self.colorMap[y, state]*15 + self.rectified[y, self.rectified.shape[1]/2])/16
                colorMapIndex[y] += 1
                colorMapIndex[y] %= self.colorMapDepth
            
            self.colorMapEmptyMedian = np.uint8(np.median(self.colorMapEmpty, axis=1))
            self.colorMapFilledMedian = np.uint8(np.median(self.colorMapFilled, axis=1))
        
        filledMask = np.zeros(self.rectCropSize[0], dtype=np.ubyte)
        emptyYMedian = np.median(self.colorMapEmptyMedian, axis=0)
        filledYMedian = np.median(self.colorMapFilledMedian, axis=0)
        distBetween, distToEmpty, distToFilled = None, None, None
        if True: #np.count_nonzero(emptyYMedian) > self.rectified.shape[0]/5 and np.count_nonzero(filledYMedian) > self.rectified.shape[0]/5:
            for y in range(self.rectCropSize[0]):
    
                imageY = self.rectified[y]
    
                if (np.count_nonzero(self.colorMapEmptyMedian[y]) != 0):
                    emptyY = self.colorMapEmptyMedian[y]
                else:
                    emptyY = emptyYMedian
                    
                if (np.count_nonzero(self.colorMapFilledMedian[y]) != 0):
                    filledY = self.colorMapFilledMedian[y]
                else:
                    filledY = filledYMedian
                
                imageY = imageY.reshape(self.rectCropSize[1]*3)
                emptyY = emptyY.reshape(self.rectCropSize[1]*3)
                filledY = filledY.reshape(self.rectCropSize[1]*3)
#                 distBetween = scipy.spatial.distance.euclidean(filledY, emptyY)
                distToEmpty = scipy.spatial.distance.euclidean(imageY, emptyY)
                distToFilled = scipy.spatial.distance.euclidean(imageY, filledY)
                    
                #triangle check
#                 if not distBetween or distToEmpty + distToFilled > distBetween*3:
#                     continue
                if distToEmpty > distToFilled:
                    filledMask[y]=255
          
            filledMask = cv2.erode(filledMask, np.ones((25),np.uint8),iterations = 1)
            filledMask = cv2.dilate(filledMask, np.ones((25),np.uint8),iterations = 1)
            c = (0, 0, 255)
            self.staticVal = self.rectified.shape[0]
            for y in range(self.rectified.shape[0]):
                if filledMask[y]:
                    if self.staticVal == self.rectified.shape[0] and filledMask[int(y + (self.rectified.shape[0]-1-y)*0.3)]:
                        self.staticVal = y
                        cv2.circle(self.rectified, (0, y), 10, c)
                        break
         
        self.imshow('colormap', np.concatenate((self.colorMapEmptyMedian*3, self.colorMapFilledMedian*3), axis=1))
        
        if self.staticVal != self.rectified.shape[0]:
            self.val = self.val*0.90 + self.staticVal*0.10
            self.levelF.write('{0}\n'.format(self.percent).encode())

        return self.val

    def rectYToVal(self, rectY):
        return 100-(rectY)*100/self.rectified.shape[0]
            
    @property
    def percent(self):
        return 100 - (self.val*100) / self.rectifiedFull.shape[0]
    
    @property
    def state(self):
        if not self.calibrated:
            return 'Calibrating'
        if not self.val:
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
                cv2.waitKey(1)
            yield tornado.gen.sleep(0.01)
        
