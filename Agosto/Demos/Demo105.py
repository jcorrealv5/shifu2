import torch, cv2
from torch import load
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import v2 as T
from PIL import Image
import numpy as np

print("Demo 105: Detectar Objetos en un Video en Tiempo Real")

transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((800, 800))
])

print("1. Crear el Modelo usando un Modelo Predefinido maskrcnn")
modelo = maskrcnn_resnet50_fpn_v2(weights=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = modelo.to(device)
modelo.eval()

print("2. Cargar los Pesos del Modelo Pre-Entrenado")
with open("C:/Users/jhonf/Documents/Shifu/Coco_maskrcnn_0.05.pt", 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

print("3. Cargar los Nombres de las Categorias de Coco")
with open("coco.names", "r") as file:
    categorias = [linea.strip() for linea in file.readlines()]

colores=[(0,0,255), (255,0,0),(0,255,0)]*30

cap = cv2.VideoCapture(0, 700)
if(cap.isOpened()):
    while True:
        rpta, img = cap.read()
        if rpta:
            imagenArray = cv2.resize(img, (800,800))
            # print("Frame: ", c)
            imagenTensor = transform(imagenArray).unsqueeze(0)
            with torch.no_grad():
                imagen = imagenTensor.view(3, 800, 800).unsqueeze(0).to(device).float()
               # salida = modelo(imagen)
            # cuadros = salida[0]["boxes"].to("cpu")
            # indicesCat = salida[0]["labels"].to("cpu")
            # nObjetosDetectados = len(cuadros)
            # for i in range(nObjetosDetectados):
            #     if(indicesCat[i]<80):
            #         c = c + 1
            #         x1=int(cuadros[i][0])
            #         y1=int(cuadros[i][1])
            #         x2=int(cuadros[i][2])
            #         y2=int(cuadros[i][3])
            #         ancho = x2 - x1
            #         alto = y2 - y1
            #         categoria = categorias[indicesCat[i]-1]
            #         #print(x1,y1)
            #         cv2.rectangle(imagenArray, rec=(x1,y1,ancho,alto), color=colores[i], thickness=3)
            #         cv2.putText(imagenArray,categoria, org=(x1,y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colores[i], thickness=5)
                
            cv2.imshow("Video de Clase", imagenArray)
            key = cv2.waitKey(0)
            if(key==ord("s")):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()