import cv2
from ultralytics import YOLO

print("Demo 119: Usando Yolo V8 para Detectar Objetos en una Imagen de Disco")
modelo = YOLO(r"C:\Users\jhonf\Documents\Shifu\yolov8s.pt")
# modelo = YOLO("yolov8s.pt")
modelo.conf = 0.30
archivo = r"C:\Users\jhonf\Documents\Shifu\shifu2\ImageNet\mesa.jpg"
results = modelo(archivo)
print("Nro resultados: ", len(results))
for result in results:
    boxes = result.boxes
    nBoxes = len(boxes)
    for i in range(nBoxes):
        indice = int(result.boxes.cls[i].item())
        nombre = result.names[indice]
        probabilidad = float(result.boxes.conf[i].item())
        print(str(i+1) + " " + nombre + ": " + str(probabilidad))
    print("result: ", type(result))
    result.show()
    result.save(filename="Cuarto_Gente.jpg")    
    imagen = cv2.imread("Cuarto_Gente.jpg")
    imagen = cv2.resize(imagen,(600,500))
    cv2.imshow("Deteccion de Objetos",imagen)
    cv2.waitKey(0)
cv2.destroyAllWindows()