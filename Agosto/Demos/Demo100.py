import torch, cv2
import torch.nn as nn
import torch.optim as optim
from torch import load
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

print("Demo 100: Probando el Modelo de Deteccion de Marcas Faciales con una Foto de Disco")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("1. Crear el Modelo tal cual fue entrenado")
class LandmarkNet(nn.Module):
    def __init__(self):
        super(LandmarkNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(in_features=7*7*1024, out_features=10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = LandmarkNet()
modelo = modelo.to(device)

print("2. Cargar los Pesos del Modelo Pre-Entrenado")
with open('CelebA_Custom_Landmarks_0.53.pt', 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

print("3. Cargar la imagen de Disco")
archivo = r"C:\Data\Python\2025_06_DADLCV\Imagenes\Alumnos\Cara2.jpg"
imagenPIL = Image.open(archivo).convert("RGB")
imagenArray = np.array(imagenPIL)
imagenOriginal = cv2.resize(imagenArray, (178,218))
imagenTensor = transform(imagenPIL).unsqueeze(0)
print("imagenTensor: ", imagenTensor.shape)

print("4. Predecir las coordenadas con la imagen de entrada")
with torch.no_grad():
    imagen = imagenTensor.view(3, 224, 224).unsqueeze(0).to(device).float()
    print("imagen: ", imagen.shape)
    salida = modelo(imagen)
    print("Salida: ", salida)
    puntosPred = salida[0].cpu().int()
    print("puntosPred: ", puntosPred)

print("5. Dibujar las coordenadas predecidas")
fig, ejes = plt.subplots()
ejes.imshow(imagenOriginal, cmap="gray")
ejes.scatter(puntosPred[::2], puntosPred[1::2], c='blue', s=10)
ejes.set_title("Imagen con Puntos Faciales Predecidos")
plt.show()