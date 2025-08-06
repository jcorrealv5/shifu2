import numpy as np
import torch, os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

class Activacion():
    def Sigmoide(x):
        return (1/(1+np.exp(-x)))
	
    def TangenteHiperbolica(x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def ReLU(x):
        return np.maximum(0, x)
    
    def LeakyRelu(alpha,x):
        return(np.maximum(alpha*x,x))

class Convolucion:
    def ObtenerSize(self,imagen, kernel, padding, stride):
        h,w = imagen.shape[-2],imagen.shape[-1]
        k_h, k_w = kernel.shape[-2],kernel.shape[-1]

        h_out = (h-k_h-2*padding)//stride[0] +1
        w_out = (w-k_w-2*padding)//stride[1] +1
        return h_out,w_out

    def Filtrar(self,imagen, kernel, bias, padding=0, stride=(1,1)):
        print("Filtrando...")
        imagenSalida = self.ObtenerSize(imagen, kernel, padding, stride)
        imagenFiltro = np.zeros(imagenSalida)
        for i in range(imagenSalida[0]):
            for j in range(imagenSalida[1]):
                imagenFiltro[i,j]=torch.tensordot(imagen[i:3+i,j:3+j],kernel).numpy() + bias.numpy()
        return imagenFiltro

class Grafico:
    def MostrarImagenes(imagenOriginal, imagenFiltro, tipoKernel):
        figura, ejes = plt.subplots(1,2)
        ejes[0].imshow(imagenOriginal, cmap="gray")
        ejes[0].set_title("Imagen Original")
        ejes[1].imshow(imagenFiltro, cmap="gray")
        ejes[1].set_title("Imagen Filtrada con Kernel " + tipoKernel)
        plt.show()

class CNN:
    def CheckAccuracy(modelo, dataLoader, device):
        num_correct = 0
        num_samples = 0
        modelo.eval()
        total = 0
        with torch.no_grad():
            for x, y in dataLoader:
                x = x.to(device)
                y = y.to(device)
                total += y.numel()
                scores = modelo(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
        modelo.train()
        print("Total de Pruebas: ", total)
        return num_correct / num_samples
    
    def CheckAccuracyBin(modelo, dataLoader, device):
        num_correct = 0
        num_samples = 0
        modelo.eval()
        with torch.no_grad():
            for x, y in dataLoader:
                x = x.to(device)
                y = y.to(device)
                scores = modelo(x)
                predictions = (torch.sigmoid(scores) > 0.5).squeeze().long()
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
        modelo.train()
        return num_correct / num_samples

    def Train(modelo, dataLoader, device, nEpocas=3, lr=0.001, stopLoss=0):
        criterio = nn.CrossEntropyLoss()
        optimizador = torch.optim.Adam(modelo.parameters(), lr)
        print("Learning Rate: ", lr)
        print("Nro Epocas: ", nEpocas)
        encontroOptimo = False
        for epoch in range(nEpocas):
            total = 0
            for id, (data, targets) in enumerate(dataLoader):
                X_train = data.to(device)
                y_train = targets.to(device)
                total += y_train.numel()
                #print(f"Epoca: {epoch}, Item: {id}, cant: {y_train.numel()}")
                scores = modelo(X_train)
                loss = criterio(scores, y_train)
                optimizador.zero_grad()
                loss.backward()
                optimizador.step()
            if(loss<=0.009 or (stopLoss>0 and loss<=stopLoss)):
                encontroOptimo = True
                torch.save(modelo.state_dict(), 'Epoca' + str(epoch) + '.pt')
                break
            print(f"Epoca: {epoch}, Perdida: {loss}, Total: {total}")
        if((stopLoss==0 and encontroOptimo==False) or encontroOptimo==False):
            torch.save(modelo.state_dict(), 'Epoca' + str(nEpocas) + '.pt')

    def TrainBin(modelo, dataLoader, device, nEpocas=3, lr=0.001, batchSize=32, totalMuestras=1000, stopLoss=0):
        criterio = nn.BCEWithLogitsLoss()
        optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)        
        print("Learning Rate: ", lr)
        print("Nro Epocas: ", nEpocas)
        print("BatchSize: ", batchSize)
        print("Total Muestras: ", totalMuestras)
        for epoch in range(nEpocas):
            total = 0
            for id, (data, targets) in enumerate(dataLoader):
                if(total<(totalMuestras-batchSize)):
                    X_train = data.to(device)
                    y_train = targets.to(device).reshape(batchSize,1).float()
                    #print(f"Epoca: {epoch}, Nro Item: {id}, y_train: {y_train}")
                    total += y_train.numel()
                    scores = modelo(X_train)
                    loss = criterio(scores, y_train)
                    optimizador.zero_grad()
                    loss.backward()
                    optimizador.step()
            if(loss<=0.009 or (stopLoss>0 and loss<=stopLoss)):
                encontroOptimo = True
                torch.save(modelo.state_dict(), 'Epoca' + str(epoch) + '.pt')
                break
            print(f"Epoca Bin: {epoch}, Perdida Bin: {loss}, Total: {total}")
        if((stopLoss==0 and encontroOptimo==False) or encontroOptimo==False):
            torch.save(modelo.state_dict(), 'Epoca' + str(nEpocas) + '.pt')

    def TrainBinLoss(modelo, dataLoader, device, criterio = nn.BCELoss(), nEpocas=3, lr=0.001, batchSize=32, totalMuestras=1000, stopLoss=0):
        optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)        
        print("Criterio: ", criterio)
        print("Learning Rate: ", lr)
        print("Nro Epocas: ", nEpocas)
        print("BatchSize: ", batchSize)
        print("Total Muestras: ", totalMuestras)
        for epoch in range(nEpocas):
            total = 0
            for id, (data, targets) in enumerate(dataLoader):
                if(total<(totalMuestras-batchSize)):
                    X_train = data.to(device)
                    y_train = targets.to(device).reshape(batchSize,1).float()
                    #print(f"Epoca: {epoch}, Nro Item: {id}, y_train: {y_train}")
                    total += y_train.numel()
                    scores = modelo(X_train)
                    loss = criterio(scores, y_train)
                    optimizador.zero_grad()
                    loss.backward()
                    optimizador.step()
            if(loss<=0.009 or (stopLoss>0 and loss<=stopLoss)):
                encontroOptimo = True
                torch.save(modelo.state_dict(), 'Epoca' + str(epoch) + '.pt')
                break
            print(f"Epoca Bin: {epoch}, Perdida Bin: {loss}, Total: {total}")
        if((stopLoss==0 and encontroOptimo==False) or encontroOptimo==False):
            torch.save(modelo.state_dict(), 'Epoca' + str(nEpocas) + '.pt')

class ConvNet2C1P2FC(nn.Module):
    def __init__(self):
        super(ConvNet2C1P2FC, self).__init__()
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.Pool = nn.MaxPool2d(2, 2)
        self.FC1 = nn.Linear(64*12*12, 128)
        self.FC2 = nn.Linear(128, 10)
        self.Dropout1 = nn.Dropout(0.25)
        self.Dropout2 = nn.Dropout(0.5)
    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Pool(x)
        x = self.Dropout1(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, 64*12*12)
        x = F.relu(self.FC1(x))
        x = self.Dropout2(x)
        x = F.relu(self.FC2(x))
        x = F.log_softmax(x, dim=1)
        return x

class ConvNetBin2C1P2FC(nn.Module):
    def __init__(self):
        super(ConvNetBin2C1P2FC, self).__init__()
        self.Conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.Pool = nn.MaxPool2d(2, 2)
        self.FC1 = nn.Linear(64*12*12, 128)
        self.FC2 = nn.Linear(128, 1)
        self.Dropout1 = nn.Dropout(0.25)
        self.Dropout2 = nn.Dropout(0.5)
    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Pool(x)
        x = self.Dropout1(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, 64*12*12)
        x = F.relu(self.FC1(x))
        x = self.Dropout2(x)
        x = self.FC2(x)
        return x

class ConvNet6C3P3FC(nn.Module):
    def __init__(self, nClases, nSize=4):
        super(ConvNet6C3P3FC, self).__init__()
        self.nSize = nSize
        self.Conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.Conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.Conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.Conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.Pool = nn.MaxPool2d(2, 2)
        self.FC1 = nn.Linear(256 * nSize * nSize, 1024)
        self.FC2 = nn.Linear(1024, 512)
        self.FC3 = nn.Linear(512, nClases)
        self.Dropout1 = nn.Dropout(0.25)
        self.Dropout2 = nn.Dropout(0.5)
    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Pool(x)
        x = F.relu(self.Conv3(x))
        x = F.relu(self.Conv4(x))
        x = self.Pool(x)
        x = F.relu(self.Conv5(x))
        x = F.relu(self.Conv6(x))
        x = self.Pool(x)
        x = self.Dropout1(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, 256 * self.nSize * self.nSize)
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = self.Dropout2(x)
        x = self.FC3(x)
        x = F.log_softmax(x, dim=1)
        return x
        
class ConvNetBinCol642C1P2FC(nn.Module):
    def __init__(self):
        super(ConvNetBinCol642C1P2FC, self).__init__()
        self.Conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.Pool = nn.MaxPool2d(2, 2)
        self.FC1 = nn.Linear(64*30*30, 128)
        self.FC2 = nn.Linear(128, 1)
        self.Dropout1 = nn.Dropout(0.25)
        self.Dropout2 = nn.Dropout(0.5)
    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = self.Pool(x)
        x = self.Dropout1(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, 64*30*30)
        x = F.relu(self.FC1(x))
        x = self.Dropout2(x)
        x = self.FC2(x)
        return x

class ConvNetBinCol643C2P3FC(nn.Module):
    def __init__(self):
        super(ConvNetBinCol643C2P3FC, self).__init__()
        self.Conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.Conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.Pool = nn.MaxPool2d(2, 2)
        self.FC1 = nn.Linear(128*8*8, 1024)
        self.FC2 = nn.Linear(1024, 128)
        self.FC3 = nn.Linear(128, 1)
        self.Dropout1 = nn.Dropout(0.25)
        self.Dropout2 = nn.Dropout(0.5)
    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = self.Pool(x)
        x = F.relu(self.Conv2(x))
        x = self.Pool(x)
        x = F.relu(self.Conv3(x))
        x = self.Pool(x)
        x = self.Dropout1(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, 128*8*8)
        x = F.relu(self.FC1(x))
        x = self.Dropout2(x)
        x = F.relu(self.FC2(x))
        x = self.FC3(x)
        x = F.sigmoid(x)
        return x

class DatasetFS(Dataset):
    def __init__(self, ruta, transform=None, separador="_", indice=0):
        self.archivos = []
        self.etiquetas = []
        self.transform = transform
        archivos = os.listdir(ruta)
        for nombreArchivo in archivos:
            archivo = os.path.join(ruta, nombreArchivo)
            self.archivos.append(archivo)
            raza = int(nombreArchivo.split(separador)[indice])
            self.etiquetas.append(raza)
    
    def __len__(self):
        return len(self.etiquetas)
        
    def __getitem__(self, indice):
        archivo = self.archivos[indice]
        imagenPIL = Image.open(archivo).convert('RGB')
        if(self.transform is not None):
            imagenTensor = self.transform(imagenPIL)
        etiqueta = self.etiquetas[indice]
        return imagenTensor, etiqueta
    
    def clases(self):
        return np.unique(self.etiquetas).tolist()