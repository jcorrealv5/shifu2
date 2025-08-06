import torch
from torch import nn,load
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
sys.path.append("../../Modulos")
from ANN import ConvNet6C3P3FC

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

print("Demo 79: Predecir Raza desde un Archivo con un Rostro")

print("1. Creando el Modelo CNN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = ConvNet6C3P3FC(5,8).to(device)

print("2. Cargar el Modelo Pre Entrenado")
with open('UTK-Face_Raza.pt', 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

rutaImagenes = "C:/Users/jhonf/Documents/Shifu/shifu2/Caras/"
archivo = rutaImagenes + "20.jpg"
imagen = Image.open(archivo).convert("RGB")
print("type imagen PIL: ", type(imagen))
imagenTensor = data_transforms(imagen).unsqueeze(0)
print("Shape Tensor: ", imagenTensor.shape)
plt.imshow(imagen, cmap="gray")
plt.show()

def mostrarRaza(etiqueta):
    if etiqueta==0:
        sexo="Blanco"
    elif etiqueta==1:
        sexo="Negro"
    elif etiqueta==2:
        sexo="Asia"
    elif etiqueta==3:
        sexo="India"
    elif etiqueta==4:
        sexo="Otros"
    return sexo

with torch.no_grad():
    imagenPlana = imagenTensor.view(3, 64, 64).to(device).float()
    print("imagenPlana: ", imagenPlana)
    print("Shape Data Prueba Final: ", imagenPlana.shape)
    salida = modelo(imagenPlana)
    print("salida: ", salida)
    _, predecido = torch.max(salida, 1)
    prediccion = predecido.item()
    print("Prediccion: ", prediccion)
    raza = mostrarRaza(prediccion)
    print("Raza: ", raza)