import cv2

# читать изображение QRCODE
img = cv2.imread("1.jpg")
# инициализируем детектор QRCode cv2
detector = cv2.QRCodeDetector()
#вы все пидоры
# обнаружhhить и декодировать
data, bbox, straight_qrcode = detector.detectAndDecode(img)
# if there is a QR code
if bbox is not None:
    print(f"QRCode data:\n{data}")
    # отображаем изображение с линиями
    # длина ограничивающей рамки
    n_lines = len(bbox)
    for i in range(n_lines):
        # рисуем все линии
        point1 = tuple(bbox[i][0])
        point2 = tuple(bbox[(i+1) % n_lines][0])
        cv2.line(img, point1, point2, color=(255, 0, 0), thickness=2)
# отобразить результат
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()