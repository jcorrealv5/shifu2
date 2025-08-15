import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import sys

data_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

inicio = datetime.now()
print("Demo 98: Entrenar un Modelo para Detectar Marcas Faciales en Personas Famosas")

print("1. Crear el DataSet CelebA")
dsTrain = datasets.CelebA(root=r"C:\Users\jhonf\Documents\Shifu\datasets",download=True,
target_type="landmarks",transform=data_transforms)
print("DataSet: ", dsTrain)

batchSize = 128
print("2. Crear el DataLoader para manejar el DataSet CelebA")
dlTrain = DataLoader(dsTrain, batch_size=batchSize, shuffle=True)
print("DataLoader: ", dlTrain)

print("3. Crear el Modelo desde ResNet con ImageNet")
clases = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

numFeatures = 512
dropout=0.2
modelo.fc = nn.Sequential(
nn.Linear(modelo.fc.in_features, numFeatures),
nn.ReLU(inplace=True),
# nn.Dropout(dropout),
nn.Linear(numFeatures,clases))

print("4. Configurar que el Modelo no calcule Pesos y Bias excepto la ultima capa")
for param in modelo.parameters():
    param.requires_grad = False
for param in modelo.fc.parameters():
    param.requires_grad = True

print("5. Configurar para que el Modelo use GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = modelo.to(device)
print(modelo)

print("6. Entrenando el Modelo en: " + device.type)
encontroOptimo=False
epocas = 10
totalMuestras = len(dsTrain)
lr = 0.001
print("Total de Muestras: ", totalMuestras)
print("Total de Epocas: ", epocas)
print("Batch Size: ", batchSize)
print("Learning Rate: ", lr)
criterio = nn.MSELoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr)
modelo.train()
for epoca in range(epocas):    
    total = 0
    for i, (entradas, etiquetas) in enumerate(dlTrain):        
        if(total<(totalMuestras-batchSize)):
            entradas, etiquetas = entradas.to(device), etiquetas.to(device).float()
            total += batchSize
            salidas = modelo(entradas)
            perdida = criterio(salidas, etiquetas)
            valor = str(round(perdida.item(),2)*100)
            print(f"Epoca: {epoca}, Nro Item: {i}, Perdida: {valor}")
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
    #scheduler.step()
    #if(perdida<=0.1):
        #encontroOptimo = True        
    torch.save(modelo.state_dict(), "CelebA_ResNet_Attr_" + valor + ".pt")
        #break
    print(f"Epoca Bin: {epoca}, Perdida Bin: {perdida}")
#if(encontroOptimo==False):
torch.save(modelo.state_dict(), "CelebA_ResNet_Attr_" + valor + ".pt")

fin = datetime.now()
tiempo = fin - inicio
print(f"7. Tiempo de Proceso: {tiempo}")