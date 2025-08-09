import torch
from torchvision import models, datasets, transforms
from torch import nn,load
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys, cv2

print("Demo 91: Prediccion del Sexo con data de validacion")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
def mostrarSexo(etiqueta):
    if etiqueta<1:
        sexo="Femenino"
    else:
        sexo="Masculino"
    return sexo

ruta = "C:/Data/Python/2025_01_PythonMV/Imagenes/DataSet/Genero/Validation"
#ruta = "C:/Data/Python/2025_06_DADLCV/Imagenes/Caras/"
print("1. Crear el DataSet de Caras UTK-Face para Pruebas")
dsTest = datasets.ImageFolder(root=ruta, transform=data_transforms)
print("DataSet Test: ", dsTest)

batchSize = 256
print("2. Crear el DataLoader para manejar el DataSet UTK-Face")
dlTest = DataLoader(dsTest, batch_size=batchSize, shuffle=True)
print("DataLoader Test: ", dlTest)

imagenes, etiquetas = next(iter(dlTest))
print("Etiquetas: ", etiquetas.shape)

print("3. Crear el Modelo desde la Red Neuronal")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = modelo.to(device)

print("4. Cargar el Modelo Pre Entrenado")
with open('UTKF_ResNet_Sexo_9.0.pt', 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))     

print("5. Cargar y Mostrar la Cara a Predecir")
imagenes, etiquetas = next(iter(dlTest))

print("6. Midiendo el Rendimiento del Modelo")
num_correct = 0
num_samples = 0
modelo.eval()
c = 0
with torch.no_grad():
    for x, y in dlTest:
        x = x.to(device)
        y = y.to(device)
        c = c + 1
        print("c: ", c)
        scores = modelo(x)
        predictions = (torch.sigmoid(scores) > 0.5).squeeze().long()
        #print("predictions: ", predictions)
        num_correct += (predictions == y).sum()
        print("num_correct: ", num_correct)        
        num_samples += predictions.size(0)
        print("num_samples: ", num_samples)        
presTrain = num_correct / num_samples
print(f"Presicion de Pruebas: {presTrain:.2f}")