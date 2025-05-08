#en sade halda 
'''
import cv2

image_path = '/home/aynur/Desktop/opencv_python/photos/highway.jpeg'
image = cv2.imread(image_path)
cv2.imshow('Image Window', image)
cv2.waitKey(0)
'''

#keyboard interrupt (ctrl + c) tutmur
'''
import cv2

image_path = '/home/aynur/Desktop/opencv_python/photos/highway.jpeg'
output_image = cv2.imread(image_path)
cv2.imshow('Output image',output_image)

#keyboard interrupt gozleyir ve yalniz esc ve ya q basildiqda pencereni baglayir ve kodu dayandirir
while True:
    key = cv2.waitKey(0) & 0xFF #keyboardi oxuyur 

    if key == ord('q') or key == 27:
        #cv2.destroyAllWindows()
        exit()
'''
# try except olmadan 100 ms gozleyerek ctrl + c ni tutur
'''
import cv2
image_path = '/home/aynur/Desktop/opencv_python/photos/highway.jpeg'
image = cv2.imread(image_path)
cv2.imshow('Image', image)
while True:
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q') or key == 27:
        print('Exiting ...')
        exit()
'''

import cv2
import os

try:
    image_path = '/home/aynur/Desktop/opencv_python/pictures/highway.jpeg'

    if not os.path.exists(image_path):
        raise FileNotFoundError("Image file does not exist!")
        #raise xetani if serti altinda tutur xetani ozu yaradir ve onu except blokuna dasiyir
        #raise goren isi sadece print ile de ede bilerik bu halda ise except islemeyecek cunki xeta handleden cixir

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Failed to load the image. The format may be unsupported or the file may be corrupted.")

    cv2.imshow('Output image', image)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            print("Exiting...")
            break

except FileNotFoundError as fnf:
    print(f"Error: {fnf}")
except ValueError as ve:
    print(f"Image loading error: {ve}")
except KeyboardInterrupt:
    print("Exited using Ctrl + C.")
finally:
    cv2.destroyAllWindows()

'''
import cv2
import os

import cv2.img_hash

image_path = '../pictures/highway.jpeg'

try:

    if not os.path.exists(image_path):
        raise FileNotFoundError()
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError()
    cv2.imshow('Image', image)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('GrayScale image', gray_scale)
    blurred_image = cv2.GaussianBlur(image, (7, 7), cv2.BORDER_ISOLATED)
    cv2.imshow('Blurred image', blurred_image)
    cannied = cv2.Canny(image, 50, 150)
    cv2.imshow('Cannied image', cannied)
    key = cv2.waitKey(100) & 0xFF
    while True:
        if key == ord('q') or key == 27:
            break
        
except ValueError as ve:
    print(f'No image!{ve}')
except FileNotFoundError as fnf:
    print(f'File is not exist!{fnf}')
except KeyboardInterrupt:
    print('Exiting by clicking Ctrl + C')
except Exception as e:
    print(f'Unexpected error occured! {e}')
finally:
    cv2.destroyAllWindows()
'''