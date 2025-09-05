import torch, cv2

modelo = torch.hub.load("ultralytics/yolov5", "yolov5s")
modelo.conf = 0.30
archivo = r"C:\Users\jhonf\Documents\Shifu\shifu2\ImageNet\mesa.jpg"
rpta = modelo(archivo)
rpta.render()
nObjetos = str(len(rpta.xyxy[0]))
imagen = rpta.ims[0]
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
print("imagen: ", imagen)
cv2.imshow("Objetos Detectados: " + nObjetos, imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()