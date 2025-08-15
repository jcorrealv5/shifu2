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

pesos = {0: 0.2255096518130976,
 1: 0.09460379928857943,
 2: 0.04858142246404975,
 3: 0.12315270935960591,
 4: 1.1130899376669634,
 5: 0.16240093542938808,
 6: 0.1044451871657754,
 7: 0.10559216083797938,
 8: 0.1041493084485919,
 9: 0.1696755802904846,
 10: 0.4767353165522502,
 11: 0.1227656648988411,
 12: 0.17470300489168414,
 13: 0.4334257975034674,
 14: 0.5441880713974749,
 15: 0.38461538461538464,
 16: 0.3950695322376738,
 17: 0.6024096385542169,
 18: 0.06502965352200603,
 19: 0.05559508984166518,
 20: 0.05968011458582,
 21: 0.05205080158234437,
 22: 0.6151574803149606,
 23: 0.2132741852926122,
 24: 0.02997961386257345,
 25: 0.08831425745372333,
 26: 0.5781683626271971,
 27: 0.09078364441862154,
 28: 0.31985670419651996,
 29: 0.39123630672926446,
 30: 0.4407616361071932,
 31: 0.05243728500713147,
 32: 0.11974327042820193,
 33: 0.0779982528391364,
 34: 0.13427865506499087,
 35: 0.5087505087505088,
 36: 0.05327309921581998,
 37: 0.20685090186993216,
 38: 0.3404139433551198,
 39: 0.0320159823784033}

inicio = datetime.now()
print("Demo 93: Entrenar un Modelo para Reconocer Personas Famosas usando CelebA")

print("1. Crear el DataSet CelebA")
dsTrain = datasets.CelebA(root="datasets",download=True,
target_type="attr",transform=data_transforms)
print("DataSet: ", dsTrain)

batchSize = 128
print("2. Crear el DataLoader para manejar el DataSet CelebA")
dlTrain = DataLoader(dsTrain, batch_size=batchSize, shuffle=True)
print("DataLoader: ", dlTrain)

print("3. Crear el Modelo desde ResNet con ImageNet")
clases = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
numFeatures = 1024
dropout=0.2
modelo.fc = nn.Sequential(nn.Flatten(),
nn.Linear(modelo.fc.in_features, numFeatures),
nn.ReLU(inplace=True),
nn.Dropout(dropout),
#nn.Linear(1024,512),
#nn.LeakyReLU(),
#nn.Dropout(0.5),
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
epocas = 100
totalMuestras = len(dsTrain)
lr = 0.001
print("Total de Muestras: ", totalMuestras)
print("Total de Epocas: ", epocas)
print("Batch Size: ", batchSize)
print("Learning Rate: ", lr)
weight_tensor = torch.tensor(list(pesos.values()), dtype=torch.float).to(device)
criterio = nn.BCEWithLogitsLoss(weight=weight_tensor)
optimizador = torch.optim.Adam(modelo.parameters(), lr)
#optimizador = torch.optim.RMSprop(modelo.parameters(), lr=lr, alpha=0.99, eps=1e-8)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizador, gamma=0.9)
for epoca in range(epocas):
    modelo.train()
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