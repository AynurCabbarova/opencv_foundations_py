import cv2
import os
import numpy as np


image_path = '../pictures/baku.jpeg'

try:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError()
    
    blank = np.zeros(image.shape[:2], dtype='uint8')
    cv2.imshow('Blank image', blank)

    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image', grayscale_image)

    cannied_image = cv2.Canny(image, 125, 175)
    cv2.imshow('Cannied image', cannied_image)

    #burdaki 125 limitdir 125 < olanlar ag, eksi olanlar qara gorunut, THRESH_BINARY_INV ise bunun eksini edir, yalni 
    # zgrayscale tetniq olunur eks halda rengliye tetbi qet yoxla
    ret, thresh = cv2.threshold(grayscale_image, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow('Tresh image', thresh)

    contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    print(f'{len(contours)} contour(s) found!')

    cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)
    cv2.imshow('Enhanced blank', blank)

    result = np.hstack([cannied_image, blank, thresh])
    cv2.imshow('Result', result)
    

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