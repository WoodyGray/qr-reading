import cv2
cam = cv2.VideoCapture(0)

result, image = cam.read()

if result:
    print(image)
    cv2.imshow("daun", image)

    cv2.imwrite("daun.png", image)

    cv2.waitKey(0)
    cv2.destroyWindow("daun")