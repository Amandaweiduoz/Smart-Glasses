import sys
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog, QPushButton, QStyle, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QGroupBox, QWidget, QPlainTextEdit
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPainterPath
from PyQt5 import QtGui
import numpy as np
import math
from utils import circles_from_p1p2r, Pt


class FaceImage(QWidget):
    width = 600

    # Initialize the coordinates of critical points in a face used to extract data
    origin_pos = [(0.45, 0.20745714285714287, 'E'),
     (0.45666666666666667, 0.3562, 'D'), (0.44666666666666666, 0.49450476190476195, 'C'),
      (0.45, 0.6654285714285715, 'F'), (0.22833333333333333, 0.39925714285714287, 'G'),
       (0.6883333333333334, 0.39795238095238095, 'K'), (0.29, 0.33662857142857144, 'Q'),
        (0.30333333333333334, 0.38751428571428576, 'I'), (0.4, 0.3927333333333333, 'J'), 
        (0.35333333333333333, 0.37316190476190475, 'H'), (0.3516666666666667, 0.4005619047619048, 'L'), 
        (0.365, 0.644552380952381, 'O'), (0.31166666666666665, 0.6158476190476191, 'M'),
     (0.5333333333333333, 0.6484666666666666, 'P'), 
     (0.5983333333333334, 0.6197619047619047, 'N')] 
    
    
    comment = [
            # top of head, top of forehead, nose, bottom of chin
            (0.5, 0.1, 'E'),   
            (0.5, 0.2, 'D'),
            (0.5, 0.3, 'C'),
            (0.5, 0.4, 'F'),

            # ears of both sides
            (0.3, 0.2, 'G'),
            (0.7, 0.2, 'K'),

            # eyebrow
            (0.4, 0.2, 'Q'),

            # eyes
            (0.22, 0.22, 'I'),
            (0.3, 0.22, 'J'),
            (0.25, 0.21, 'H'),
            (0.25, 0.23, 'L'),


            # points to describe the curve of chin
            (0.4, 0.35, 'O'),
            (0.3, 0.3, 'M'),
            (0.6, 0.35, 'P'),
            (0.7, 0.3, 'N')

    ]
    def __init__(self, path, onUpdate=None):
        super(QWidget, self).__init__()
        # self.setFixedSize(600, 400)
        self.setFixedSize(self.width, self.width)
        self.p = None
        self.chosen_points = [
        ]
        self.selected_label = None
        self.onUpdate = onUpdate
        self.showGlass = False
        self.points = {}
        if path:
            self.p = QPixmap(path)
            # self.p.scaledToHeight(600)
            # self.p.scaledToWidth(400)
            # self.setScaledContents(True)
            # self.setPixmap(self.p)

            w = self.p.size().width()
            h = self.p.size().height()
            print(self.p.size(), w, h)
            ww = self.width
            wh = self.width / w * h
            self.wh = wh
            self.ww = ww
            self.setFixedSize(ww, wh)
            self.add_points()
        p = self.pos()
        print(dir(p), p.x(), p.y())

    def doPoints(self):
        self.showGlass = False
        self.update()
    
    def saveImg(self):
        p = QPixmap.grabWindow(self.winId())
        p.save('output', 'png')

    def add_points(self):
        self.points = {
            label: QPoint(int(w*self.ww), int(h*self.wh)) for (w, h, label) in self.origin_pos
        }
        self.rule_of_points('E', self.points['E'])
        self.rule_of_points('G', self.points['G'])

    def drawCross(self, p, v, size): 
        vx1 = QPoint(v.x() - size, v.y())
        vx2 = QPoint(v.x() + size, v.y())
        vy1 = QPoint(v.x(), v.y() + size)
        vy2 = QPoint(v.x(), v.y() - size)
        p.drawLine(vx1, vx2)
        p.drawLine(vy1, vy2)

    def paintEvent(self, paint_event):
        painter = QtGui.QPainter(self)
        if self.p:
            painter.drawPixmap(self.rect(), self.p)
        pen = QtGui.QPen()
        pen.setColor(QColor('red'))
        pen.setWidth(1)
        painter.setPen(pen)
        pen2 = QtGui.QPen()
        pen2.setColor(QColor('black'))
        pen.setWidth(1)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        # painter.drawPoint(300, 300)
        # painter.drawLine(100, 100, 400, 400)

        for k, v in self.points.items():
            # print(pos)
            # if k == self.selected_label:
            #     painter.setPen(pen2)
            # p = painter.drawPoint(v)
            if not self.showGlass:
                self.drawCross(painter, v, 10 if k == self.selected_label else 5)
                painter.drawText(QPoint(v.x() + 5, v.y() - 5), k)

        if self.points and not self.showGlass:
            painter.setPen(pen2)
            painter.drawPolyline(*[self.points[k] for k in ['E', 'D', 'C', 'F']])
            painter.drawPolyline(*[self.points[k] for k in ['G', 'K']])
            # painter.setPen(pen)
            # print(p)
        if self.showGlass:
            for k, v in self.glass.items():
                print(k, v)
                # self.drawCross(painter, v, 5)
                # painter.drawText(QPoint(v.x() + 5, v.y() - 5), k)
            # painter.drawPolyline(*[self.glass[k] for k in ['rp2', 'rp3', 'rp5', 'rp4', 'rp2']])
            pen2 = QtGui.QPen()
            pen2.setColor(QColor('black'))
            pen2.setWidth(0.2/self.ratio)
            painter.setPen(pen2)
            for f in [self.drawUp, self.drawDown, self.drawLeft, self.drawRight, 
                    self.drawUpRight, self.drawDownRight, self.drawLeftRight, self.drawRightRight]:
                x, y, r, start, stop = f()
                print(f, x, y, r, start, stop)
                painter.drawArc(x, y, r, r, start * 16, (stop - start)*16 )
                self.drawNoise(painter)
                painter.drawLine(self.glass['p2'], self.points['G'])
                painter.drawLine(self.glass['rp2'], self.points['K'])
            # x2, y2, r2, r22 = self.drawDown()
            # painter.drawArc(x2, y2, r2, r2, 0 * 16, 360 * 16)

            # x3, y3, r3, r33 = self.drawLeft()
            # print('Left', x3, y3, r3)
            # painter.drawArc(x3, y3, r3, r3, 0 * 16, 360 * 16)

            # x4, y4, r4, _ = self.drawRight()
            # painter.drawArc(x4, y4, r4, r4, 0 * 16, 360 * 16)
        self.onUpdate and self.onUpdate()

    def drawUp(self):
        r = 1/self.g['g9']/self.ratio
        p2 = self.glass['p2']
        x = self.glass['p1'].x()
        y = np.sqrt(r**2 - (x - p2.x())**2) + p2.y()
        print(x, y, r)
        # start = 90  - np.pi
        c = QPoint(x, y)
        return x - r, y - r, r*2, self.calcAngle(self.glass['p3'], c, 0), self.calcAngle(self.glass['p2'], c, 0)

    
    def drawUpRight(self):
        r = 1/self.g['g9']/self.ratio
        p2 = self.glass['rp2']
        x = self.glass['rp1'].x()
        y = np.sqrt(r**2 - (x - p2.x())**2) + p2.y()
        print(x, y, r)
        # start = 90  - np.pi
        c = QPoint(x, y)
        return x - r, y - r, r*2, self.calcAngle(self.glass['rp3'], c, 0), self.calcAngle(self.glass['rp2'], c, 0)

    @staticmethod
    def calcAngle(c, p, offset):
        '''
        with c as the origin, calculate the angle (in degree) between point p and -x-axis, 
        '''
        # angle = np.arcsin((p.y() - c.y())/r)/np.pi * 180
        angle = np.arctan2(p.y() - c.y(), p.x() - c.x()) * 180 / np.pi
        if offset != 0:
            angle = offset - angle
        return angle

    def drawDown(self):
        r = 1/self.g['g10']/self.ratio
        x = self.glass['p1'].x()
        p4 = self.glass['p4']
        y = -np.sqrt(r**2 - (x - p4.x())**2) + p4.y()
        c = QPoint(x, y)
        return x - r, y - r, 2*r, self.calcAngle(self.glass['p4'], c, 0), self.calcAngle(self.glass['p5'], c, 0)

    def drawDownRight(self):
        r = 1/self.g['g10']/self.ratio
        x = self.glass['rp1'].x()
        p4 = self.glass['rp4']
        y = -np.sqrt(r**2 - (x - p4.x())**2) + p4.y()
        c = QPoint(x, y)
        return x - r, y - r, 2*r, self.calcAngle(self.glass['rp4'], c, 0), self.calcAngle(self.glass['rp5'], c, 0)

    def drawLeftRight(self):
        '''
        left side frame of the right part of glasses
        '''
        r = 1/self.g['g11']/self.ratio
        c1, c2 = self.get_circle(self.glass['rp2'], self.glass['rp4'], r)
        c = c1 if c1.x < self.glass['rp2'].x() else c2
        x = c.x
        y = c.y
        c = QPoint(x, y)
        return x - r, y - r, 2*r, self.calcAngle(self.glass['rp2'], c, 180), self.calcAngle(self.glass['rp4'], c, -180)

    def drawNoise(self, painter):
        p3 = self.glass['p3']
        p5 = self.glass['p5']
        rp3 = self.glass['rp3']
        rp5 = self.glass['rp5']
        path = QPainterPath()
        path.moveTo(p3)
        y0 = p3.y() + (p5.y() - p3.y())/5
        x0 = (p3.x() + rp3.x())/2
        cp = QPoint(x0, y0)
        path.cubicTo(p3, cp, rp3)
        # path.cubicTo
        painter.drawPath(path)

    def drawLeft(self):
        r = 1/self.g['g11']/self.ratio
        
        y = np.abs((self.glass['p2'].y() + self.glass['p4'].y())/2)
        p2 = self.glass['p2']
        p4 = self.glass['p4']
        # x =  np.sqrt(r**2 - (y - p2.y())**2) + p2.x()
        p2_ = Pt(p2.x(), p2.y())
        p4_ = Pt(p4.x(), p4.y())
        c1, c2 = circles_from_p1p2r(p2_, p4_, r)
        if c1.x > p2.x():
            c = c1
        else:
            c = c2
        c = QPoint(c.x, c.y)
        x = c.x()
        y = c.y()
        return x - r, y - r, 2*r, self.calcAngle(self.glass['p2'], c, 180), self.calcAngle(self.glass['p4'], c, 180)
    
        
    def drawRightRight(self):
        '''
        right side frame of the right part of glasses
        '''
        r = 1/self.g['g12']/self.ratio
        c1, c2 = self.get_circle(self.glass['rp3'], self.glass['rp5'], r)
        c = c1 if c1.x > self.glass['rp3'].x() else c2
        x = c.x
        y = c.y
        c = QPoint(x, y)
        offset = 180
        return x - r, y - r, 2*r, self.calcAngle(self.glass['rp5'], c, 180), self.calcAngle(self.glass['rp3'], c, offset)

    def get_circle(self, p1, p2, r):
        p1_ = Pt(p1.x(), p1.y())
        p2_ = Pt(p2.x(), p2.y())
        c1, c2 = circles_from_p1p2r(p1_, p2_, r)
        return c1, c2

    def drawRight(self):
        r = 1/self.g['g12']/self.ratio
        p3 = self.glass['p3']
        p5 = self.glass['p5']
        p3_ = Pt(p3.x(), p3.y())
        p5_ = Pt(p5.x(), p5.y())
        c1, c2 = circles_from_p1p2r(p3_, p5_, r)
        print(c1, c2)
        c = c1 if c1.x < p3.x() else c2
        x = c.x
        y = c.y
        c = QPoint(x, y)
        offset = 180
        return x - r, y - r, 2*r,  self.calcAngle(self.glass['p5'], c, -180), self.calcAngle(self.glass['p3'], c, offset)
    
    def calcGlass(self):
        self.glass = {}
        g = self.points['G']
        d = self.points['D']
        q = self.points['Q']
        x1 = (g.x() + d.x())/2
        y1 = q.y()
        self.glass['p1'] = QPoint(x1, y1)
        length = self.g['g1'] * self.g['g2'] / self.ratio
        self.glass['p2'] = QPoint(x1 - length / 2, y1)
        self.glass['p3'] = QPoint(x1 + length / 2, y1)


        y4 = y1 + self.g['g7']/self.ratio
        ydiff = y4 - y1
        a4 = self.g['g3']
        # a4 = 90 - (self.g['g4'] - 90)
        x4 = np.sqrt((ydiff/np.sin(a4/180*np.pi))**2 - ydiff**2) + self.glass['p2'].x()
        self.glass['p4'] = QPoint(int(x4), y4)
        x5 =  np.abs(np.sqrt((ydiff/np.sin(self.g['g5']/180*np.pi))**2 - ydiff**2) - self.glass['p3'].x())
        # print(x5, np.sqrt((ydiff/np.sin(self.g['g5']/180*np.pi))**2 - ydiff**2), self.glass['p3'].x(), ydiff)
        self.glass['p5'] = QPoint(int(x5), y4)

        ## calculate the coordinates of the points on the left side by symmetric
        x0 = self.points['E'].x()
        self.glass['rp1'] = self.mirrorPoint(x0, self.glass['p1'])
        self.glass['rp2'] = self.mirrorPoint(x0, self.glass['p2'])
        self.glass['rp3'] = self.mirrorPoint(x0, self.glass['p3'])
        self.glass['rp4'] = self.mirrorPoint(x0, self.glass['p4'])
        self.glass['rp5'] = self.mirrorPoint(x0, self.glass['p5'])

    def mirrorPoint(self, x0, p):
        return QPoint(2*x0 - p.x(), p.y())

    def drawGlass(self):
        self.calcGlass()
        self.showGlass = True
        self.update()
        pass

    def mouseReleaseEvent(self, cursor_event):
        # self.chosen_points.append(cursor_event.pos())
        # # self.chosen_points.append(self.mapFromGlobal(QtGui.QCursor.pos()))
        # self.update()
        self.selected_label = None
        pass

    def mousePressEvent(self, cursor_event):
        pos = cursor_event.pos()
        limit = 5
        for k, v in self.points.items():
            if abs(v.x() - pos.x()) < limit and abs(v.y() - pos.y()) < limit:
                self.selected_label = k
                print(k)
                return
        self.selected_label = None
        print(pos)

    def mouseMoveEvent(self, cursor_event):
        print(cursor_event)
        if self.selected_label:
            print(self.selected_label)
            p = self.points[self.selected_label]
            self.do_repaint(cursor_event)
            self.update()

    def rule_of_points(self, label, p):
        v = ('E', 'D', 'C', 'F')
        h = ('G', 'K')
        if label in v:
            for k in v:
                self.points[k] = QPoint(p.x(), self.points[k].y())
        if label in h:
            for k in h:
                self.points[k] = QPoint(self.points[k].x(), p.y())


    def do_repaint(self, e):
        pos = e.pos()
        self.points[self.selected_label] = QPoint(pos)
        self.rule_of_points(self.selected_label, self.points[self.selected_label])

    def output(self):
        size = (self.ww, self.wh)
        r = [(v.x(), v.y(), k) for k, v in self.points.items()]
        print(r, size)
        return size, r

    def get_dist(self, a, b, axios='y'):
        pa = self.points[a]
        pb = self.points[b]
        if axios == 'y':
            l = pa.y() - pb.y()
        else:
            l = pa.x() - pb.x()
        return abs(l) * self.ratio

    @staticmethod
    def define_circle(p1, p2, p3):
        """
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return (None, np.inf)

        # Center of circle
        cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        radius = math.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        print('Define Circle', radius, cx, cy)
        return ((cx, cy), radius)

    @staticmethod
    def angle(A, B, C):
        '''
        calculate the angle ABC
        '''
        a = np.radians(np.array(A))
        b = np.radians(np.array(B))
        c = np.radians(np.array(C))
        u = a - b
        v = c - b
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def calc(self):
        '''
        calculate the dimensions of the face needed for the math model using coordinates extracted above
        '''
        self.ratio = 15/(self.points['K'].x() - self.points['G'].x())
        # version 3 ratio
        # self.ratio = 30/(self.ww)
        f1 = self.get_dist('E', 'D', 'y')
        f2 = self.get_dist('D', 'C', 'y')
        f3 = self.get_dist('G', 'I', 'x')
        f4 = self.get_dist('H', 'L', 'y')
        i = self.points['I']
        h = self.points['H']
        j = self.points['J']
        l = self.points['L']

        f5 = self.define_circle((i.x(), i.y()), (h.x(), h.y()), (j.x(), j.y()))[1]*self.ratio

        f6 = self.define_circle((i.x(), i.y()), (l.x(), l.y()), (j.x(), j.y()))[1]*self.ratio

        to_p = lambda label: (self.points[label].x(), self.points[label].y())
        m = to_p('M')
        o = to_p('O')
        f = to_p('F')
        p = to_p('P')
        n = to_p('N')

        f7 = self.angle(m, o, f) + self.angle(f, p, n)
        f7 = f7 / 2

        g1 = 4.41701-0.08044 * f1 + 0.13534 * f3 - 0.22261 * f4 - 0.00574 * f6
        g2 = 1.01480 + 0.03911 * f1 + 0.10595 * f4 + 0.00350 * f6
        g3 = 89.25338 + 1.01041 * f1 + 0.65124 * f2 - 1.59652 * f3 - 0.48595 * f5 + 0.04303 * f6
        g4 = 113.35996 + 0.61537 * f2 - 0.71446 * f3
        g5 = 84.41795 + 1.16988 * f1 + 1.22221 * f2 - 2.14174 * f3 - 0.58859 * f5 + 0.04562 * f6
        g7 = 2.89302 + 0.07672 * f1 + 0.13167 * f2 - 0.04477 * f5 + 0.00556 * f6
        g8 = 0.53853 + 0.01556 * f1 + 0.02530 * f2 - 0.00865 * f5 + 0.00105 * f6
        g9 = -0.04068 + 0.00602 * f1 + 0.01929 * f4 + 0.00052669 * f6
        g10 = -0.02861 + 0.00681 * f1 + 0.03020 * f4 + 0.00062614 * f6
        g11 = -0.15427 + 0.00066656 * f7 + 0.01065 * f1 + 0.04094 * f4 + 0.00091191 * f6
        g12 = -0.05294 + 0.01204 * f1 + 0.05344 * f4 + 0.00108 * f6

        # version 3 params
        g1 = 4.41-0.080 * f1+0.13 * f3-0.22 * f4-0.0057 * f6
        g2 = 1.01+0.039 * f1+0.11 * f4+0.0035 * f6
        g3 = 89.25+1.01 * f1+0.65 * f2-1.59 * f3-0.48 * f5+0.043 * f6
        g6 = 118.23-0.63 * f3+0.84 * f4
        g7 = 2.89+0.077 * f1+0.13 * f2-0.044 * f5+0.0056 * f6
        g9 = 0.059+0.0060 * f1+0.019 * f4+0.00053 * f6
        g10 = 0.071+0.0068 * f1+0.030 * f4+0.00063 * f6
        g11 = -0.05+0.00067 * f7+0.011 * f1+0.041 * f4+0.00091 * f6
        g12 = 0.047+0.012 * f1+0.053 * f4+0.0011 * f6

        # version 5 params
        g1 = 4.41-0.080 * f1+0.13 * f3-0.22 * f4-0.0057 * f6
        g2 = 1.01 + 0.039 * f1+0.11 * f4+0.0035 * f6 
        g3 = 89.25+1.01 * f1+0.65 * f2-1.59 * f3-0.48 * f5+0.043 * f6
        g6 = 118.23-0.63 * f3+0.84 * f4
        g7 = 2.89+0.077 * f1+0.13 * f2-0.044 * f5+0.0056 * f6
        g9 = 0.039+0.0060 * f1+0.019 * f4+0.00053 * f6
        g10 = 0.051+0.0068 * f1+0.030 * f4+0.00063 * f6
        g11 = -0.07+0.00067 * f7+0.011 * f1+0.041 * f4+0.00091 * f6
        g12 = 0.027+0.012 * f1+0.053 * f4+0.0011 * f6
        f = {
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'f4': f4,
            'f5': f5,
            'f6': f6,
            'f7': f7
        }
        g = {
            'g1': g1,
            'g2': g2,
            'g3': g3,
            'g4': g4,
            'g5': g5,
            'g7': g7,
            'g8': g8,
            'g9': g9,
            'g10': g10,
            'g11': g11,
            'g12': g12

        }
        self.g = g
        return f, g


class FaceContainer(QGroupBox):
    def __init__(self, path):
        super(QGroupBox, self).__init__('眼镜编辑器')
        self.root = QHBoxLayout()
        self.img = FaceImage(path)
        self.root.addWidget(self.img)
        self.panel = QGroupBox()
        self.root.addWidget(self.panel)
        self.setLayout(self.root)
        # self.panel.SetFixedSize(600, 200)
        self._panel()
    
    def openFile(self):
        path = QFileDialog.getOpenFileName(window, "Open")[0]
        print(path, type(path))

        # self.root.removeWidget(self.img)
        if path:
            img = FaceImage(path, self.show)
            self.root.replaceWidget(self.img, img)
            del self.img
            self.img = img
            self.show()
        # self.root.addWidget(self.img)

    def show(self):
        size, ps = self.img.output()
        text = 'size of photo： {}(width)x{}(height)\n\n'.format(size[0], int(size[1]))
        i = 0
        for x, y, k in ps:
            text += '{}: {}, {}      {}'.format(k, x, y, '\n' if i%2 else '')
            i += 1
        # text += '\n' + '-'*100
        self.text.setPlainText(text)

        f, g = self.img.calc()
        text = 'dimensions：\n'
        for k, v in f.items():
            text += '{}: {}\n'.format(k, v)

        text += '\n'
        for k, v in g.items():
            text += '{}: {}\n'.format(k, v)
        self.calc.setPlainText(text)

    def saveImg(self):
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow(self.winId())
        screenshot.save("screenshot.png", 'png')

    def _panel(self):
        layout = QVBoxLayout()
        self.clear = QPushButton('move the points')
        self.save = QPushButton('save photo')
        self.back = QPushButton('open')
        self.gen = QPushButton('generate glasses')
        self.text = QPlainTextEdit()
        self.text.setFixedHeight(200)
        self.text.setDisabled(True)

        self.calc = QPlainTextEdit()
        self.calc.setFixedHeight(400)
        self.calc.setDisabled(True)
        
        self.back.clicked.connect(self.openFile)
        self.gen.clicked.connect(lambda : self.img.drawGlass())
        self.clear.clicked.connect(lambda : self.img.doPoints())
        self.save.clicked.connect(lambda : self.saveImg())
        self.buttons = [
            # self.clear,
            self.back,
            self.gen,
            self.clear,
            self.save,
            self.text,
            self.calc
        ]
        for b in self.buttons:
            layout.addWidget(b)
        self.panel.setLayout(layout)

# Subclass QMainWindow to customise your application's main window
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Smart Glasses")
        self.setMinimumSize(800, 800)
        # label = QLabel("This is a PyQt5 window!")
        play_button = QPushButton('Choose Photo')
        # play_button.setIcon(QApplication.style().standardIcon(QStyle.SP_FileIcon))
        play_button.clicked.connect(self.openFile)
        # play_button.setMaximumSize(100, 50)
        self.open_button = play_button
        # The `Qt` namespace has a lot of attributes to customise
        # widgets. See: http://doc.qt.io/qt-5/qt.html
        # play_button.setAlignment(Qt.AlignCenter)
        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        # self.start()
        fc = FaceContainer('')
        self.setCentralWidget(fc)

    def start(self):
        self.setCentralWidget(self.open_button)

    def openFile(self):
        path = QFileDialog.getOpenFileName(window, "Open")[0]
        print(path, type(path))
        self.label = FaceContainer(path)
        self.setCentralWidget(self.label)



# app = QApplication(sys.argv)
appctxt = ApplicationContext()
window = MainWindow()
window.show()
exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
sys.exit(exit_code)
# app.exec_()
