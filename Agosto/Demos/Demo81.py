import cv2, sys
from PIL import Image
import torch
from torch import nn,load
import torchvision.transforms as transforms
sys.path.append("../../Modulos")
from ANN import ConvNet6C3P3FC

def mostrarRaza(etiqueta):
    if etiqueta==0:
        raza="Blanco"
    elif etiqueta==1:
        raza="Negro"
    elif etiqueta==2:
        raza="Asia"
    elif etiqueta==3:
        raza="India"
    elif etiqueta==4:
        raza="Otros"
    return raza

print("Demo 81: Detectar la Raza en un Video en Tiempo Real")
cap = cv2.VideoCapture(0, 700)
if(cap.isOpened()):
    c = 0
    #Crear un Clasificador para Reconocer Rostros usando Haar Cascade
    archivoHaar = "haarcascade_frontalface_default.xml"
    clasificador = cv2.CascadeClassifier(archivoHaar)    
    #Crear el Modelo CNN para Clasificar Sexo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = ConvNet6C3P3FC(5,8).to(device)
    #Cargar los Pesos del Modelo Pre Entrenado
    with open('UTK-Face_Raza.pt', 'rb') as f: 
        modelo.load_state_dict(load(f, map_location=device, weights_only=True))
        modelo.eval()
    data_transforms = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])            
    while True:
        rpta, imagen = cap.read()
        if(rpta):
            c = c + 1
            #Detectar las caras usando el Clasificador
            caras = clasificador.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=5, minSize=(50,50),flags=cv2.CASCADE_SCALE_IMAGE)
            nCaras = len(caras)
            #Si existe al menos una cara detectada cargar el Modelo
            if(nCaras>0):
                for(x,y,w,h) in caras:
                    #Dibujar un Rectangulo en la Cara
                    cv2.rectangle(imagen, (x,y), (x+w, y+h), (0,255,0), 3)
                    cara = imagen[y:y+h,x:x+w]
                    imagenPIL = Image.fromarray(cara).convert("RGB")
                    imagenTensor = data_transforms(imagenPIL).unsqueeze(0)
                    with torch.no_grad():
                        imagenPlana = imagenTensor.view(3, 64, 64).to(device).float()
                        print("Shape Data Prueba Final: ", imagenPlana.shape)
                        salida = modelo(imagenPlana)
                        _, predecido = torch.max(salida, 1)
                        prediccion = predecido.item()
                        raza = mostrarRaza(prediccion)
                        cv2.putText(imagen, raza, org=(x,y-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=5)

            cv2.imshow("Video WebCam", imagen)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("La Camara Web No esta Activa")