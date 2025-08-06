import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append("../../Modulos")
from ANN import DatasetFS, CNN, ConvNet6C3P3FC

inicio = datetime.now()
print("Demo 77: Crear una CNN para Clasificacion Categorica Raza 64x64")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])
ruta = "C:/Data/Python/2025_06_DADLCV/DataSets/UTKFace/train"
print("1. Crear el DataSet de Caras UTK-Face")
dsTrain = DatasetFS(ruta,data_transforms,"_",2)
print("Etiquetas: ", dsTrain.clases())

batchSize = 512
print("2. Crear el DataLoader para manejar el DataSet UTK-Face")
dlTrain = DataLoader(dsTrain, batch_size=batchSize, shuffle=True)
print("DataLoader Train: ", dlTrain)

imagenes, etiquetas = next(iter(dlTrain))
imagenTensor = imagenes[0]
etiquetaTensor = etiquetas[0]

print("3. Crear el Modelo desde la Red Neuronal")
modelo = ConvNet6C3P3FC(5,8).to(device)

print("4. Entrenando el Modelo en: " + device.type)
nArchivos = len(dsTrain)
CNN.Train(modelo, dlTrain, device, nEpocas=100, lr=0.001, stopLoss=0)

print("5. Midiendo el Rendimiento del Modelo")
presTrain = CNN.CheckAccuracy(modelo, dlTrain, device)
print(f"Presicion del Entrenamiento: {presTrain:.2f}")

print("6. Guardando el Modelo")
torch.save(modelo.state_dict(), 'UTK-Face.pt')

fin = datetime.now()
tiempo = fin - inicio
print(f"7. Tiempo de Proceso: {tiempo}")