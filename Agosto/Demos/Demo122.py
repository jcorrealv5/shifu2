import cv2
from ultralytics import YOLO

print("Demo 122: Usando Yolo V8 para Segmentar Objetos en una Imagen de Disco")
modelo = YOLO("yolov8s-seg.pt")
modelo.conf = 0.70
archivo = r"C:\Users\jhonf\Documents\Shifu\shifu2\ImageNet\varios.jpg"
results = modelo(archivo)
result = results[0]
imagen = result.plot()
imagen = cv2.resize(imagen,(500,500))
cv2.imshow("Segmentacion de Objetos",imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()