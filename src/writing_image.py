import cv2

input_path = '/home/aynur/Desktop/opencv_python/pictures/highway.jpeg'
image = cv2.imread(input_path, 1)
#for grayscale use 0

output_path = '/home/aynur/Desktop/opencv_python/pictures/highway_2.jpeg'
output_image = cv2.imwrite(output_path, image)