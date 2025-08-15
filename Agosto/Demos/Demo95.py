import torch
from torch import nn,load
from torchvision import models, transforms
import matplotlib.pyplot as plt
import sys
from PIL import Image

data_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

print("Demo 95: Predecir Multiples Caracteristicas en una Foto con un Rostro")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. Crear el Modelo desde ResNet con ImageNet")
modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
numFeatures = 1024
dropout=0.2
clases = 40
modelo.fc = nn.Sequential(nn.Flatten(),
nn.Linear(modelo.fc.in_features, numFeatures),
nn.ReLU(inplace=True),
nn.Dropout(dropout),
nn.Linear(numFeatures,clases))
modelo = modelo.to(device)

print("2. Cargar el Modelo Pre Entrenado")
with open('CelebA_ResNet_Attr_4.0.pt', 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

print("3. Plotear la Imagen a Clasificar")
rutaImagenes = "C:/Data/Python/2025_06_DADLCV/Imagenes/Alumnos/"
archivo = rutaImagenes + "Edelson.jpg"
imagen = Image.open(archivo).convert("RGB")
print("type imagen PIL: ", type(imagen))
imagenTensor = data_transforms(imagen).unsqueeze(0)
print("Shape Tensor: ", imagenTensor.shape)
plt.imshow(imagen, cmap="gray")
plt.show()

with open("CelebA_Atributos_Castellano.txt","r") as file:
    atributosNombres = [linea.strip() for linea in file.readlines()]

print("3. Mostrar los Atributos de la Imagen")
with torch.no_grad():
    imagenPlana = imagenTensor.view(3, 64, 64).unsqueeze(0).to(device)
    salida = modelo(imagenPlana)
    print("Salida: ", salida)
    for i,rpta in enumerate(salida[0]):
        valor = "Si" if rpta>0.5 else "No"
        print(f"{atributosNombres[i]}: {valor}")