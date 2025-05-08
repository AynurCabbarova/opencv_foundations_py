import cv2

image_path = '/home/aynur/Desktop/opencv_python/pictures/highway.jpeg'
image = cv2.imread(image_path)

#converting grayscale
'''
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray)
'''

#blur image
'''
blur = cv2.GaussianBlur(image, (7, 7), cv2.BORDER_REFLECT)
#use (a, b) odd numbers pair a can be different from or equal to b
cv2.imshow('Blurred', blur)
'''

#distinguish original and blurred image
'''
import numpy as np
if image is None:
    print('Image is not exist!')
else:
    blurred_image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    
    combined = np.hstack((image, blurred_image, image))#combines images like stack
    
    cv2.imshow('Original vs Blurred', combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#edge cascade, for the best result use (50, 150)
'''
canny = cv2.Canny(image, 50, 150)
canny_ = cv2.Canny(image, 0, 10)
cv2.imshow('Canny_', canny_)
cv2.imshow('Canny', canny)
'''

#drawing contours on orijinal image
'''
if image is None:
    print('No image found!')
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 125, 150)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.RETR_LIST all contours
    #cv2.RETR_EXTERNAL only external contours
    #cv2.RETR_TREE both all contours and their hierarchy relations

    #cv2.CHAIN_APPROX_NONE all points of contour
    #cv2.CHAIN_APPROX_SIMPLE only points that lie on a line

    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    #-1 represents draw all contours, we can use spesific id but i want to see all of these contours

    cv2.imshow('Contours Window', image)

    cv2.imwrite('../pictures/contour_output.jpeg', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''

#dilate, merge contours
'''
canny = cv2.Canny(image, 50, 150)
dilated = cv2.dilate(canny, (3, 3), iterations=2)
cv2.imshow('Canny', canny)
cv2.imshow('Dilated', dilated)
'''

#eroding
'''
canny = cv2.Canny(image, 50, 150)
dilated = cv2.dilate(canny, (7, 7), iterations=1)
eroded = cv2.erode(dilated, (7, 7), iterations=1)
cv2.imshow('Eroded', eroded)
'''

#resizing
'''
frame = cv2.resize(image, (500, 500))
cv2.imshow('Resized', frame)
'''

#cropping
'''
cropped = image[10:200, 300:350]
cv2.imshow("Cropped", cropped)
'''

#blurring cropped part and adding original image or blurring spesific part of image
'''
cv2.imshow('Real image', image)
cropped = image[100:200, 200:300]
cropped = cv2.GaussianBlur(cropped, (7, 7), cv2.BORDER_DEFAULT)
image[100:200, 200:300] = cropped
cv2.imshow('Image', image)
'''

#flipping
'''
flipped = cv2.flip(image, -1)
#0 for vertically, 1 for horizontally, -1 before horizontally, then vertically
cv2.imshow('Flipped image', flipped)
'''

cv2.waitKey(0)
