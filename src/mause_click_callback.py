import cv2

image_path = '../pictures/highway.jpeg'
image = cv2.imread(image_path)

if image is None:
    print("No image!")
    exit()

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 0, 255), cv2.FILLED)
        
        coord_text = f"({x},{y})"
        cv2.putText(image, coord_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

cv2.namedWindow('Interactive Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Interactive Image', click_event)

while True:
    cv2.imshow('Interactive Image', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
