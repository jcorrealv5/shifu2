import cv2
from ultralytics import YOLO

print("Demo 123: Usando Yolo V8 para Segmentar Objetos en una Video de Disco")
modelo = YOLO("yolov8s-seg.pt")
modelo.conf = 0.30
archivo = r"C:\Users\jhonf\Documents\Shifu\test_person.mp4"
cap = cv2.VideoCapture(archivo)
if(cap.isOpened()):
    while True:
        rpta, img = cap.read()
        if rpta:
            results = modelo(img)
            imagen = results[0].plot()
            imagen = cv2.resize(imagen,(500,500))
            cv2.imshow("Video de Disco",imagen)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
cap.release()
cv2.destroyAllWindows()