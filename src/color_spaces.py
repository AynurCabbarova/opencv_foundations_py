import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


image_path = '../pictures/baku.jpeg'

try:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError()
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_to_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_to_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    edited = cv2.cvtColor(lab, cv2.COLOR_HSV2BGR)
    edited_ = cv2.cvtColor(hsv, cv2.COLOR_LAB2BGR)

    result = np.hstack([image, hsv, lab])
    cv2.imshow('Result', result)
    cv2.imshow('Inverted hsv', hsv_to_bgr)
    cv2.imshow('Inverted lab', lab_to_bgr)
    cv2.imshow('Edited', edited)
    cv2.imshow('Edition 2', edited_)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('RGB', rgb)
    
    plt.imshow(rgb)
    plt.show()
    
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