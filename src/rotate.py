import cv2
import numpy as np


def myGetRotationMatrix2D(center, angle, scale):
    theta = np.deg2rad(angle)
    alpha = np.cos(theta) * scale 
    beta = np.sin(theta) * scale 

    (x, y) = center
    Matt = np.float32([
        [alpha, beta, (1 - alpha) * x - beta * y], 
        [-beta,  alpha, beta * x + (1 - alpha) * y]
    ])
    return Matt

def rotateImage(image, angle, rotPoint=None):
    (height, width) = image.shape[:2]
    scale = 2.0

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    #rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)

    #not with ready function
    rotMat = myGetRotationMatrix2D(rotPoint, angle, scale)

    theta = np.deg2rad(angle)
    cos = np.abs(np.cos(theta) * scale)
    sin = np.abs(np.sin(theta) * scale)

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    dimensions = (new_width, new_height)

    rotMat[0, 2] += (new_width / 2) - rotPoint[0]
    rotMat[1, 2] += (new_height / 2) - rotPoint[1]

    return cv2.warpAffine(image, rotMat, dimensions)

image_path = '../pictures/highway.jpeg'
image = cv2.imread(image_path)

#rotated = rotateImage(image, 45, (0, 0))
rotated = rotateImage(image, 45)

cv2.imshow('Rotated', rotated)
cv2.imshow('Original Image', image)

rrotated = rotateImage(rotated, -45)
#cv2.imshow('Double Rotated', rrotated)#maragli bir sey olacaq :)
cv2.waitKey(0)


#crop, rotate, and combine
'''
import cv2
import os
import numpy as np

def rotateImage(image, angle, rotPoint=None):
    if rotPoint is None:
        (h, w) = image.shape[:2]
        h //= 2
        w //= 2
        rotPoint = (h, w)

        dimensions = (w, h)

        rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)

        return cv2.warpAffine(image, rotMat, dimensions)

        
image_path = '../pictures/highway.jpeg'
try:
    if not os.path.exists(image_path):
        raise FileNotFoundError()
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError()
    
    cropped = image[100:200, 200:300]
    cropped = rotateImage(cropped, 90)

    cropped = cv2.resize(cropped, (image.shape[1], image.shape[0]))

    result = np.hstack((image, cropped))
    cv2.imshow('Result', result)
    cv2.waitKey(0)

except FileNotFoundError as fnf:
    print(f'File path is not correct{fnf}.')
except ValueError as ve:
    print(f'Unable to open the image! {ve}')
except KeyboardInterrupt:
    print('Exitin by CTRL + C.')
except Exception as e:
     print(f'Unexpected error occured!{e}')
finally:
    cv2.destroyAllWindows()
'''