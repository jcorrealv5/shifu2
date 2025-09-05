import torch, cv2
from torchvision import transforms

print("Demo 107: Usando Yolo V5 con un Video de Disco")
modelo = torch.hub.load("ultralytics/yolov5", "yolov5s")
modelo.conf = 0.30
archivo = r"http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"

transform = transforms.Compose([transforms.ToTensor()])

cap = cv2.VideoCapture(archivo)
if(cap.isOpened()):
    while True:
        rptaFrame, imagen = cap.read()
        if rptaFrame:
            rpta = modelo(imagen)
            rpta.render()
            nObjetos = str(len(rpta.xyxy[0]))
            imagenBox = rpta.ims[0]
            cv2.imshow("Deteccion de Objetos en un Video", imagenBox)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
