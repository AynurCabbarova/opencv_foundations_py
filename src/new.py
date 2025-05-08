import cv2
import os 
import sys
import numpy as np


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_dir = os.path.join(base_dir, 'my_modules')
sys.path.append(target_dir)
from draw import *


class CanvasAnalyzer:
    def __init__(self, width = 800, height = 800):
        self.original_canvas = np.zeros([width, height, 3], dtype='uint8')
        self.analyzer_canvas = np.zeros([width, height, 3], dtype='uint8')


    def draw_circle_(self):
        draw_circle(self.original_canvas, (60, 60), 20, (255, 0, 255))
        put_text(self.original_canvas, 'Circle: 1', (30, 30), (255, 0, 255))
        draw_circle(self.original_canvas, (100, 240), 40, (0, 255, 255))
        put_text(self.original_canvas, 'Circle: 2', (60, 190), (0, 255, 255))
        draw_circle(self.original_canvas, (300, 300), 30, (255, 255, 0))
        put_text(self.original_canvas, 'Circle: 3', (270, 260), (255, 255, 0))

    
    def draw_rectangle_(self):
        draw_rectangle(self.original_canvas, (125, 540), (525, 590), (0, 255, 0))
        put_text(self.original_canvas, 'Rectangle: 1', (125, 530), (0, 255, 0))
        draw_rectangle(self.original_canvas, (400, 100), (500, 200), (0, 0, 255))
        put_text(self.original_canvas, 'Rectangle: 2', (400, 80), (0, 0, 255))


    def draw_line_(self):
        draw_line(self.original_canvas, (340, 400), (580, 400), (255, 0, 0))
        put_text(self.original_canvas, 'Line: 1', (340, 390), (255, 0, 0))


    def process_image(self):
        grayscale_image = cv2.cvtColor(self.original_canvas, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grayscale_image, 50, 255, cv2.THRESH_BINARY)
        return thresh
    

    def draw_contours(self, thresh):
        contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contoured_image(self.analyzer_canvas, contours, -1, (0, 255, 0))
        return contours


    def contour_analyzing(self, contours):
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            if area > 1000:
                draw_rectangle(self.analyzer_canvas, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 255, 0))
                put_text(self.analyzer_canvas, f'Area : {area:.0f}', (x - 5, y + h + 20), (255, 255, 255))
            else:
                draw_rectangle(self.analyzer_canvas, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255))


    def showing_canvases(self):
        cv2.imshow('Original Canvas', self.original_canvas)
        cv2.imshow('Analysis Canvas', self.analyzer_canvas)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 27:
                print('Exiting ...')
                break
    

    def reset_canvas(self):
        self.analyzer_canvas[:] = 0
        self.original_canvas[:] = 0


if __name__ == '__main__':
    analyzer = CanvasAnalyzer()
    analyzer.draw_circle_()
    analyzer.draw_rectangle_()
    analyzer.draw_line_()
    thresh = analyzer.process_image()
    contours = analyzer.draw_contours(thresh)
    analyzer.contour_analyzing(contours)
    status = analyzer.showing_canvases()
