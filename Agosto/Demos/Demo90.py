import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime

inicio = datetime.now()
print("Demo 90: Entrenar un Modelo usando Transfer Learning con Caracteristicas Fijas para Detectar Sexo")

print("1. Crear el Modelo con la Arquitectura ResNet usando los Pesos de ImageNet")
modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

modelo.fc = nn.Sequential(nn.Flatten(),
nn.Linear(modelo.fc.in_features, 1024),
nn.ReLU(),
nn.Dropout(0.2),
nn.Linear(1024,512),
nn.ReLU(),
nn.Dropout(0.2),
nn.Linear(512,1),
nn.Sigmoid())

print("2. Configurar que el Modelo no calcule Pesos y Bias excepto la ultima capa")
for param in modelo.parameters():
    param.requires_grad = False
for param in modelo.fc.parameters():
    param.requires_grad = True

print("3. Configurar para que el Modelo use GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = modelo.to(device)
print(modelo)

data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

ruta = "C:/Data/Python/2025_01_PythonMV/Imagenes/DataSet/Genero/Training"
print("4. Crear el DataSet de Caras UTK-Face")
dsTrain = datasets.ImageFolder(root=ruta, transform=data_transforms)
print("DataSet Train: ", dsTrain)

batchSize = 32
print("5. Crear el DataLoader para manejar el DataSet UTK-Face")
dlTrain = DataLoader(dsTrain, batch_size=batchSize, shuffle=True)
print("DataLoader Train: ", dlTrain)

print("6. Entrenar el Modelo congelando las capas de caracteristicas y solo entrando el clasificador")
encontroOptimo=False
epocas = 10
totalMuestras = len(dsTrain)
lr = 0.01
print("Total de Muestras: ", totalMuestras)
print("Total de Epocas: ", epocas)
print("Batch Size: ", batchSize)
print("Learning Rate: ", lr)
criterio = nn.BCELoss()
#optimizador = optim.Adam(modelo.fc.parameters(), lr=lr)
optimizador = optim.SGD(modelo.fc.parameters(), lr=lr, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizador, gamma=0.9)
modelo.train()
for epoca in range(epocas):    
    total = 0
    for i, (entradas, etiquetas) in enumerate(dlTrain):        
        if(total<(totalMuestras-batchSize)):
            entradas, etiquetas = entradas.to(device), etiquetas.to(device).reshape(batchSize,1).float()
            print(f"Epoca: {epoca}, Nro Item: {i}")
            total += etiquetas.numel()
            optimizador.zero_grad()
            salidas = modelo(entradas)
            perdida = criterio(salidas, etiquetas)
            valor = str(round(perdida.item(),2)*100)
            perdida.backward()
            optimizador.step()
    scheduler.step()
    if(perdida<=0.1):
        #encontroOptimo = True        
        torch.save(modelo.state_dict(), "UTKF_ResNet_Sexo_" + valor + ".pt")
        #break
    print(f"Epoca Bin: {epoca}, Perdida Bin: {perdida}")
#if(encontroOptimo==False):
torch.save(modelo.state_dict(), "UTKF_ResNet_Sexo_" + valor + ".pt")

fin = datetime.now()
tiempo = fin - inicio
print(f"7. Tiempo de Proceso: {tiempo}")