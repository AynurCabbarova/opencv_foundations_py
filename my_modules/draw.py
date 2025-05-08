import cv2
import numpy as np


def draw_circle(image, center_point, R, color):
    cv2.circle(image, center_point, R, color, 1)

def draw_rectangle(image, start_point, end_point, color):
    cv2.rectangle(image, start_point, end_point, color, 1)

def draw_line(image, start_point, end_point, color):
    cv2.line(image, start_point, end_point, color, 5)

def put_text(image, content, point, color):
    cv2.putText(image, content, point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1, cv2.LINE_AA)

def contour_detection(image):
    return cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

def contoured_image(image, contours, contourIdx, color):
    cv2.drawContours(image, contours, contourIdx, color)