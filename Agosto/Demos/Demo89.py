import os
import torch
from torchvision import models
from PIL import Image

print("Demo 89: Clasificacion Automatica de Objetos en Archivos de un Directorio")
ruta = input("Ingresa la ruta con las imagenes a clasificar: ")
if(os.path.isdir(ruta)):
    carpeta = os.path.basename(ruta)
    print("Directorio: " , carpeta)
    archivos = os.listdir(ruta)
    with open("imagenet_classes.txt", "r") as file:
        clases = [linea.strip() for linea in file.readlines()]
    pesos = models.ResNet18_Weights.IMAGENET1K_V1
    modelo = models.resnet18(pesos)
    modelo.eval()
    rpta = ""
    for nombreArchivo in archivos:
        tipo = nombreArchivo.split(".")[-1].lower()
        if(tipo=="jpg" or tipo=="jpeg" or tipo=="png"):
            print("Clasificando archivo: ", nombreArchivo)
            archivo = os.path.join(ruta, nombreArchivo)
            imagen = Image.open(archivo).convert('RGB')
            preProceso = pesos.transforms()
            imagenProcesada = preProceso(imagen)
            entrada = imagenProcesada.unsqueeze(0)           
            with torch.no_grad():
                salida = modelo(entrada)
            probabilidades = torch.nn.functional.softmax(salida[0], dim=0)
            top_prob, top_catid = torch.topk(probabilidades, 1)
            rpta += nombreArchivo + " => " + clases[top_catid] + " - " + str(round(top_prob.item()*100,2)) + "%" + "\n"
    archivoRpta = carpeta + ".txt"
    with open(archivoRpta, "w") as file:
        file.write(rpta)
    print(f"El archivo de informe: {archivoRpta} fue creado.")
else:
    print("La ruta No existe")