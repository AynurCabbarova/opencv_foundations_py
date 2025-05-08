#draving rectangle around largest contour 
'''
import cv2
import os


image_path = '../pictures/highway.jpeg'

try:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError()
    
    cannied = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 125, 175)
    contours, _ = cv2.findContours(cannied, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key = cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    area = cv2.contourArea(largest_contour)
    cv2.putText(image, f'Area : {area:.0f}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Image', image)

    #drawing the largest contour on object not rectangle
    #cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 1)
    
    
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


#drawing rectangles around all contours
'''
import cv2
import os


cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

image_path = '../pictures/highway.jpeg'

try:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError()
    
    cannied = cv2.Canny((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), 125, 175)
    contours, _ = cv2.findContours(cannied, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 100.0:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
    cv2.imshow('Image', image)
    
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


#hierarchies
'''
import cv2
import os
import numpy as np


image_path = '../pictures/highway.jpeg'

try:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError()
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image', grayscale_image)

    cannied_image = cv2.Canny(image, 125, 175)
    cv2.imshow('Cannied image', cannied_image)

    contours, hierarchies = cv2.findContours(cannied_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for i in range(len(contours)):
        if hierarchies[0][i][2] == -1:
            color = (0, 255 - 20*i, 20*i % 255)
            cv2.drawContours(image, contours, i, color, 1)
            #cv2.drawContours(image, contours, i, (0, 255, 0), 1)
    cv2.imshow('Original', image)

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