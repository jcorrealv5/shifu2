import torch
from torchvision import models, datasets, transforms
from torch import nn,load
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys, cv2

print("Demo 94: Prediccion de Atributos de Celebridades con data de validacion")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

print("1. Crear el DataSet CelebA")
dsVal = datasets.CelebA(root="datasets",download=True,
target_type="attr",transform=data_transforms)
print("DataSet: ", dsVal)

batchSize = 128
print("2. Crear el DataLoader para manejar el DataSet CelebA")
dlVal = DataLoader(dsVal, batch_size=batchSize, shuffle=False)
print("DataLoader: ", dlVal)

print("3. Crear el Modelo desde ResNet con ImageNet")
modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
numFeatures = 1024
dropout=0.2
clases = 40
modelo.fc = nn.Sequential(nn.Flatten(),
nn.Linear(modelo.fc.in_features, numFeatures),
nn.ReLU(inplace=True),
nn.Dropout(dropout),
nn.Linear(numFeatures,clases))

print("4. Configurar que el Modelo no calcule Pesos y Bias excepto la ultima capa")
for param in modelo.parameters():
    param.requires_grad = False
for param in modelo.fc.parameters():
    param.requires_grad = True

modelo = modelo.to(device)
print(modelo)

print("5. Cargar el Modelo Pre Entrenado")
with open('CelebA_ResNet_Attr_4.0.pt', 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

print("6. Cargar y Mostrar la Cara a Predecir")
imagenes, etiquetas = next(iter(dlVal))
imagenTensor, etiquetaTensor = imagenes[0], etiquetas[0]
print("Shape Tensor Prueba: ", imagenTensor.shape)
print("Shape Tensor Salida: ", etiquetaTensor.shape)

imagenArray = imagenTensor.permute(1, 2, 0).numpy()
etiqueta = etiquetaTensor.detach().numpy()
print("Shape Array Prueba: ", imagenArray.shape)
plt.imshow(imagenArray, cmap="gray")
plt.show()

with open("CelebA_Atributos_Castellano.txt","r") as file:
    atributosNombres = [linea.strip() for linea in file.readlines()]    

print("6. Usar el Modelo para Clasificar el Objeto")
with torch.no_grad():
    imagenPlana = imagenTensor.view(3, 64, 64).unsqueeze(0).to(device)
    #print("imagenPlana: ", imagenPlana)
    #print("Shape Data Prueba Final: ", imagenPlana.shape)
    salida = modelo(imagenPlana)
    print("Salida: ", salida)
    for i,rpta in enumerate(salida[0]):
        valor = "Si" if rpta>0.5 else "No"
        print(f"{atributosNombres[i]}: {valor}")    

print("7. Midiendo el Rendimiento del Modelo")
num_correct = 0
num_samples = 0
modelo.eval()
with torch.no_grad():
    c = 0
    for x, y in dlVal:
        c = c + 1
        x = x.to(device)
        y = y.to(device)
        scores = modelo(x)
        predictions = (scores > 0.5).float()
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)
        print(f"{c}, num_samples: {num_samples}")
    precision = num_correct / num_samples
    print(f"Precision: {precision}")