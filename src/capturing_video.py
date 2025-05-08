#cameradan goruntu almaq
'''
import cv2

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    
    if not ret:
        print('Exiting ...')
        break

    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

    cv2.imshow('Live stream', frame)

video.release()
cv2.destroyAllWindows()
'''

#taking screenshot
'''
import cv2

video = cv2.VideoCapture(0)

cv2.namedWindow('Video frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video frame', 500, 500)

try:
    if not video.isOpened():
        raise ValueError()
    
    while True:
        ret, frame = video.read()

        if not ret:
            print('Video ended!')
            break

        cv2.imshow('Video frame', frame)
        key = cv2.waitKey(100) & 0xFF

        if key == 27 or key == ord('q'):
            print('Exiting...')
            break
        elif key == ord('s') or key == ord('S'):
            print('Screenshot was taken!')
            cv2.imwrite('../pictures/ss.jpeg', frame)
except ValueError as ve:
    print(f'Unable to open the camera! {ve}')
except KeyboardInterrupt:
    print('Exitin by CTRL + C.')
except Exception:
    print('Unexpected error occured!')
finally:
    video.release()
    cv2.destroyAllWindows()
'''


'''
import cv2

video_path = '/home/aynur/Desktop/opencv_python/videos/highway.mp4'
video = cv2.VideoCapture(video_path)

while True:
    check, frame = video.read()

    if not check:
        print('Video ended ...')
        break

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord('q'):
        print('Exiting ...')
        break

    cv2.imshow('Video Window', frame)

video.release()
cv2.destroyAllWindows()
'''

'''
import cv2 

video_path = '/home/aynur/Desktop/opencv_python/videos/highway.mp4'
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print('Video could not be opened!')
    exit()

cv2.namedWindow('Video window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video window', 800, 600)

while True:
    ret, frame = video.read()
    if not ret:
        print('Video finished!')
        break

    #cv2.error: OpenCV(4.11.0) /io/opencv/modules/highgui/src/window.cpp:973: error: 
    # (-215:Assertion failed) size.width>0 && size.height>0 in function 'imshow' qarsini alir 
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    print(f'FPS: {fps:.2f}, Frame: {frame_num}')

    cv2.putText(frame, f'Frame: {frame_num}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video window', frame) #diqqet label eyni olanda capture ve cv2 window eyni pencerede eks halda musteqil olaraq ayri ayri pencerelerde yayimlanir

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        print('Keyboard Interrupted')
        break

video.release()
cv2.destroyAllWindows()
'''

import cv2
import os

video = None

try:
    video_path = '/home/aynur/Desktop/opencv_python/videos/highway.mp4'

    if not os.path.exists(video_path):
        raise FileNotFoundError(f'File not found: {video_path}')
    
    cv2.namedWindow('Video Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Window', 800, 600)
    
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError('Could not open video file.')
    while True:
        ret, frame = video.read()

        if not ret:
            print('Video finished!')
            break

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        #though it is int we convert to int because of .0

        print(f'FPS: {fps}, Frame count : {frame_num}')

        #simple text
        #cv2.putText(frame, f'Frame : {frame_num}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 3)

        # White shadow(just change bgr)
        '''
        cv2.putText(frame, f'Frame num : {frame_num}', (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)  # shadow layer
        cv2.putText(frame, f'Frame num : {frame_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # text layer
        '''

        #colorful rectangle
        (text_w, text_h), _ = cv2.getTextSize(f"Frame: {frame_num}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        #cv2.getTextSize(F"Frame: {frame_num}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)  returnes 2 value  first of them is tuple contains text_width and text_height
        # the second one is baseline
        # in brief (text_size), baseline 

        #in one line eroding image frame for contour detection
        cv2.imshow('Contour detection', cv2.erode(cv2.dilate(cv2.Canny(frame, 300, 300), (3, 3), iterations=1), (3, 3), iterations=1))

        # -1: fully filled rectangle, 0: thinnest contour (not valid here), 1, 2, 3...: contour gets thicker as the value increases
        cv2.rectangle(frame, (10, 5), (20 + text_w, 10 + text_h + 10), (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {frame_num}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Video Window', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            print('Exiting...')
            break

except FileNotFoundError as fnf:
    print(f'Error: {fnf}')
except ValueError as ve:
    print(f'Error: {ve}')
    exit()
except Exception as err:
    print(f'Unexpected error occured: {err}')
finally:
    if video is not None and video.isOpened():
        video.release()
    cv2.destroyAllWindows()