import cv2 

def rescaleImages(frame, scale=0.5):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
    #kiçildirsənsə, INTER_AREA
    #böyüdürsənsə, onda INTER_LINEAR və ya INTER_CUBIC


image_path = '/home/aynur/Desktop/opencv_python/pictures/highway.jpeg'
image = cv2.imread(image_path)

image_resized = rescaleImages(image)

cv2.imshow('Original Image Frame', image)
cv2.imshow('Resized Image Frame', image_resized)

cv2.waitKey(0)

#rescale image with warpAffine 
'''
import cv2
import numpy as np

def rescaleImage(image, scale):
    height = int(image.shape[0] * scale)
    width = int(image.shape[1] * scale)

    rescaleMat = np.float32([[scale, 0, 0], [0, scale, 0]])

    dimensions = (width, height)

    return cv2.warpAffine(image, rescaleMat, dimensions)

image_path = '/home/aynur/Desktop/opencv_python/pictures/highway.jpeg'
image = cv2.imread(image_path)

image_resized = rescaleImage(image, 2)

cv2.imshow('Original Image Frame', image)
cv2.imshow('Resized Image Frame', image_resized)

cv2.waitKey(0)
'''


#rescaling videos
'''
import cv2

def rescaleFrame(frame, scale = 0.5):
    heigth = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (width, heigth)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Video', 800, 600)
cv2.namedWindow('Resized Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resized Video', 400, 300)

video_path = '/home/aynur/Desktop/opencv_python/videos/highway.mp4'
video = cv2.VideoCapture(video_path)

while True:
    ret, frame = video.read()

    if not ret:
        print('Video ended!')
        break

    frame_resized = rescaleFrame(frame)

    cv2.imshow('Original Video', frame)
    cv2.imshow('Resized Video', frame_resized)

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
'''