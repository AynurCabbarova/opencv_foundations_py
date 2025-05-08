import cv2
import numpy as np
import os
import sys


base_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
target_dir = os.path.join(base_dir, 'my_modules')
sys.path.append(target_dir)

from draw import *

try:
    width, height = 800, 800
    original_canvas = np.zeros((height, width, 3), dtype='uint8')
    analysis_canvas = np.zeros((height, width, 3), dtype='uint8')

    draw_circle(original_canvas, (60, 60), 20, (255, 0, 255))
    put_text(original_canvas, 'Circle: 1', (40, 30), (255, 0, 255))
    draw_circle(original_canvas, (100, 240), 40, (0, 255, 255))
    put_text(original_canvas, 'Circle: 2', (80, 190), (0, 255, 255))
    draw_circle(original_canvas, (300, 300), 30, (255, 255, 0))
    put_text(original_canvas, 'Circle: 3', (280, 260), (255, 255, 0))
    draw_rectangle(original_canvas, (125, 540), (525, 590), (0, 255, 0))
    put_text(original_canvas, 'Rectangle: 1', (125, 530), (0, 255, 0))
    draw_rectangle(original_canvas, (400, 100), (500, 200), (0, 0, 255))
    put_text(original_canvas, 'Rectangle: 2', (400, 80), (0, 0, 255))
    draw_line(original_canvas, (340, 400), (580, 400), (255, 0, 0))
    put_text(original_canvas, 'Line: 1', (340, 390), (255, 0, 0))

    gray = cv2.cvtColor(original_canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = contour_detection(thresh)
    contoured_image(analysis_canvas, contours, -1, (0, 255, 0))
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > 9000:
            draw_rectangle(analysis_canvas, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 255, 255))
            put_text(analysis_canvas, f'Area : {area:.2f}', (x, y + h + 20), (255, 255, 255))
            center_x, center_y = x + w // 2, y + h // 2
            if abs(center_x - 400) < 10 and abs(center_y - 400) < 10:
                region = 'Center'
            elif center_x < 400 and center_y < 400:
                region = 'Top Left'
            elif center_x < 400 and center_y > 400:
                region = 'Bottom Left'
            elif center_x > 400 and center_y < 400:
                region = 'Top Right'
            else:
                region = 'Bottom Right'
            put_text(analysis_canvas, f'Region : {region}', (x, y + h + 50), (255, 255, 255))
        elif area > 50:
            draw_rectangle(analysis_canvas, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 0))

    cv2.imshow('Original Canvas', original_canvas)
    cv2.imshow('Analysis Canvas', analysis_canvas)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 27 or key == ord('q'):
            print("Exiting...")
            break
except KeyboardInterrupt:
    print("Exited with Ctrl + C")
except Exception as e:
    print(f"Unexpected error occurred: {e}")
finally:
    cv2.destroyAllWindows()
