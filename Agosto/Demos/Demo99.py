import torch, cv2
import torch.nn as nn
import torch.optim as optim
from torch import load
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


print("Demo 99: Probando el Modelo de Deteccion de Marcas Faciales con Data de Prueba")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("1. Cargando el DataSet y el DataLoader de Test")
dsTest = datasets.CelebA(root="C:/Users/jhonf/Documents/Shifu/datasets",download=True,
target_type="landmarks",split="test",transform=transform)
dlTest = torch.utils.data.DataLoader(dsTest, batch_size=32, shuffle=True)
totalMuestras = len(dsTest)
print("Total de Muestras Test: ", totalMuestras)

print("2. Cargar el Primer Lote (Batch) y obtener la primera entrada y salida")
imagenes, puntos = next(iter(dlTest))
print("Nro de Imagenes cargadas: ", len(imagenes))
imagenTensor, puntos = imagenes[0], puntos[0]
print("Imagen Tensor: ", imagenTensor.shape)
print("Primera salida: ", puntos)

print("3. Dibujar la primera imagen con sus puntos")
imagenArray = imagenTensor.permute(1, 2, 0).numpy()
imagenOriginal = cv2.resize(imagenArray, (178,218))
print("Imagen Array: ", imagenArray.shape)
fig, ejes = plt.subplots()
ejes.imshow(imagenOriginal, cmap="gray")
ejes.scatter(puntos[::2], puntos[1::2], c='red', s=10)
ejes.set_title("Imagen con Puntos Faciales Reales")
plt.show()

print("4. Crear el Modelo tal cual fue entrenado")
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

print("5. Cargar los Pesos del Modelo Pre-Entrenado")
with open('CelebA_Custom_Landmarks_0.53.pt', 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

print("6. Predecir las coordenadas con la imagen de entrada")
with torch.no_grad():
    imagen = imagenTensor.view(3, 224, 224).unsqueeze(0).to(device)
    salida = modelo(imagen)
    print("Salida: ", salida)
    puntosPred = salida[0].cpu().int()
    print("puntosPred: ", puntosPred)

print("7. Dibujar las coordenadas predecidas")
fig, ejes = plt.subplots()
ejes.imshow(imagenOriginal, cmap="gray")
ejes.scatter(puntosPred[::2], puntosPred[1::2], c='blue', s=10)
ejes.set_title("Imagen con Puntos Faciales Predecidos")
plt.show()