import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append("../../Modulos")
from ANN import CNN, ConvNet6C3P3FC

data_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

inicio = datetime.now()
print("Demo 93: Entrenar un Modelo para Reconocer Personas Famosas usando CelebA")

print("1. Crear el DataSet CelebA")
dsTrain = datasets.CelebA(root="datasets",download=True,
target_type="identity",transform=data_transforms)
print("DataSet: ", dsTrain)

batchSize = 32
print("2. Crear el DataLoader para manejar el DataSet CelebA")
dlTrain = DataLoader(dsTrain, batch_size=batchSize, shuffle=True)
print("DataLoader: ", dlTrain)

print("3. Crear el Modelo desde la Red Neuronal")
clases = 10177
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = ConvNet6C3P3FC(clases).to(device)

print("4. Entrenando el Modelo en: " + device.type)
nArchivos = len(dsTrain)
CNN.Train(modelo, dlTrain, device, nEpocas=10, lr=0.001, stopLoss=0)

fin = datetime.now()
tiempo = fin - inicio
print(f"7. Tiempo de Proceso: {tiempo}")