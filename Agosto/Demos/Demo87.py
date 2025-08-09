import torch
from torchvision import models
from PIL import Image

print("Demo 87: Clasificar Imagenes usando la Arquitectura PreDefinida ResNet con el DataSet ImageNet Pre-Entrenado")
pesos = models.ResNet18_Weights.IMAGENET1K_V1
modelo = models.resnet18(pesos)
print(modelo)

archivo = r"C:\Data\Python\2025_06_DADLCV\Imagenes\CIFAR10\Venado.jpeg"
imagen = Image.open(archivo).convert('RGB')

preProceso = pesos.transforms()
imagenProcesada = preProceso(imagen)
print("Shape imagenProcesada: ", imagenProcesada.shape)
entrada = imagenProcesada.unsqueeze(0)
print("Shape entrada: ", entrada.shape)

modelo.eval()
with torch.no_grad():
    salida = modelo(entrada)
    #print("Salida: ", salida)

probabilidades = torch.nn.functional.softmax(salida[0], dim=0)
#print("probabilidades: ", probabilidades)

top_prob, top_catid = torch.topk(probabilidades, 1)
print(f"top_prob: {top_prob} , top_catid: {top_catid}")

with open("imagenet_classes.txt", "r") as file:
    clases = [linea.strip() for linea in file.readlines()]
print(f"Prediccion: {clases[top_catid]} - Probabilidad: {top_prob.item()}")