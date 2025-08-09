import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime

inicio = datetime.now()
print("Demo 90: Entrenat un Modelo usando Transfer Learning con Caracteristicas Fijas para Detectar Sexo")

print("1. Crear el Modelo con la Arquitectura ResNet usando los Pesos de ImageNet")
modelo = models.resnet18(pretrained=True)
modelo.fc = nn.Linear(modelo.fc.in_features, 2)

print("2. Configurar que el Modelo no calcule Pesos y Bias excepto la ultima capa")
for param in modelo.parameters():
    param.requires_grad = False
modelo.fc.weight.requires_grad = True
modelo.fc.bias.requires_grad = True

print("3. Configurar para que el Modelo use GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = modelo.to(device)
#print(modelo)

data_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

ruta = "C:/Data/Python/2025_01_PythonMV/Imagenes/DataSet/Genero/Training"
print("4. Crear el DataSet de Caras UTK-Face")
dsTrain = datasets.ImageFolder(root=ruta, transform=data_transforms)
print("DataSet Train: ", dsTrain)

batchSize = 512
print("5. Crear el DataLoader para manejar el DataSet UTK-Face")
dlTrain = DataLoader(dsTrain, batch_size=batchSize, shuffle=True)
print("DataLoader Train: ", dlTrain)

print("6. Entrenar el Modelo congelando las capas de caracteristicas y solo entrando el clasificador")
encontroOptimo=False
epocas = 100
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.fc.parameters(), lr=0.00001)
for epoca in range(epocas):
    modelo.train()
    for entradas, etiquetas in dlTrain:
        entradas, etiquetas = entradas.to(device), etiquetas.to(device)
        optimizador.zero_grad()
        salidas = modelo(entradas)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()
    if(perdida<=0.009):
        encontroOptimo = True
        torch.save(modelo.state_dict(), 'UTKF_ResNet_Sexo.pt')
        break
    print(f"Epoca Bin: {epoca}, Perdida Bin: {perdida}")
if(encontroOptimo==False):
    torch.save(modelo.state_dict(), 'UTKF_ResNet_Sexo2.pt')

fin = datetime.now()
tiempo = fin - inicio
print(f"7. Tiempo de Proceso: {tiempo}")