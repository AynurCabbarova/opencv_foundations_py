import cv2
import numpy as np

def translation(image, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (image.shape[1], image.shape[0])
    return cv2.warpAffine(image, transMat, dimensions)

image_path = '../pictures/highway.jpeg'
image = cv2.imread(image_path)

translated = translation(image, 10, -20)

cv2.imshow('Translated', translated)
cv2.waitKey(0)

#complex task
'''
import cv2
import numpy as np
import os

def rotateImage(image, angle, rotPoint):
    if rotPoint is None:
        (h, w) = image.shape[:2]
        rotPoint = (w // 2, h // 2)

    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 0.5)

    dimension = (w, h)
    return cv2.warpAffine(image, rotMat, dimension)

def translateImage(image, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])

    (h, w) = image.shape[:2]
    dimensions = (w, h)

    return cv2.warpAffine(image, transMat, dimensions)

image_path = '../pictures/highway.jpeg'

try:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError()
    
    image = translateImage((rotateImage(image, 90, None)), 100, 50)
    cv2.imshow('Image', image)
    cv2.imwrite('../pictures/transformed_output.jpg', image)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 27 or key == ord('q'):
            print('Exiting ...')
            break
    

except FileNotFoundError as fnf:
    print(f'Invalid path! {fnf}')
except ValueError as ve:
    print(f'Image is not found! {ve}')
except KeyboardInterrupt:
    print(f'Keyboard interrupted, exiting by clicking Ctrl + C!')
except Exception as e:
    print(f'Unexpected error occured! {e}')
finally:
    cv2.destroyAllWindows()
'''