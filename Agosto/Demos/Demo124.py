import cv2
from ultralytics import YOLO

print("Demo 124: Usando Yolo V8 para Segmentar Objetos en una Video en Tiempo Real")
modelo = YOLO("yolov8s-seg.pt")
modelo.conf = 0.30
cap = cv2.VideoCapture(0, 700)
if(cap.isOpened()):
    while True:
        rpta, img = cap.read()
        if rpta:
            results = modelo(img)
            imagen = results[0].plot()
            cv2.imshow("Video en Tiempo Real",imagen)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
cap.release()
cv2.destroyAllWindows()