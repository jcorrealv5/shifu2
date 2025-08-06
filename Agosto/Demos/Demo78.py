import torch
from torchvision import datasets, transforms
from torch import nn,load
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys, cv2
sys.path.append("../../Modulos")
from ANN import DatasetFS, CNN, ConvNet6C3P3FC

print("Demo 72: Prediccion de la Raza con data de validacion")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])
    
def mostrarRaza(etiqueta):
    if etiqueta==0:
        sexo="Blanco"
    elif etiqueta==1:
        sexo="Negro"
    elif etiqueta==2:
        sexo="Asia"
    elif etiqueta==3:
        sexo="India"
    elif etiqueta==4:
        sexo="Otros"
    return sexo

ruta = "C:/Data/Python/2025_06_DADLCV/DataSets/UTKFace/test"
print("1. Crear el DataSet de Caras UTK-Face para Pruebas")
dsTest = DatasetFS(ruta,data_transforms,"_",2)
print("DataSet Test: ", dsTest)

batchSize = 512
print("2. Crear el DataLoader para manejar el DataSet UTK-Face")
dlTest = DataLoader(dsTest, batch_size=batchSize, shuffle=True)
print("DataLoader Test: ", dlTest)

imagenes, etiquetas = next(iter(dlTest))
print("Etiquetas: ", etiquetas.shape)

print("3. Crear el Modelo desde la Red Neuronal")
modelo = ConvNet6C3P3FC(5,8).to(device)

print("4. Cargar el Modelo Pre Entrenado")
with open('UTK-Face_Raza.pt', 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

print("5. Cargar y Mostrar la Cara a Predecir")
imagenes, etiquetas = next(iter(dlTest))
imagenTensor, etiquetaTensor = imagenes[0], etiquetas[0]
print("Shape Tensor Prueba: ", imagenTensor.shape)
print("Shape Tensor Salida: ", etiquetaTensor.shape)

imagenArray = imagenTensor.permute(1, 2, 0).numpy()
etiqueta = etiquetaTensor.detach().numpy()
print("Shape Array Prueba: ", imagenArray.shape)
plt.imshow(imagenArray, cmap="gray")
plt.show()

print("6. Usar el Modelo para Clasificar el Objeto")
with torch.no_grad():
    imagenPlana = imagenTensor.view(3, 64, 64).to(device)
    print("imagenPlana: ", imagenPlana)
    print("Shape Data Prueba Final: ", imagenPlana.shape)
    salida = modelo(imagenPlana)
    print("Salida: ", salida)
    _, predecido = torch.max(salida, 1)
    prediccion = predecido.item()
    print("Prediccion: ", mostrarRaza(prediccion))

print("7. Midiendo el Rendimiento del Modelo")
presTest = CNN.CheckAccuracy(modelo, dlTest, device)
print(f"Presicion de Pruebas: {presTest:.2f}")