from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt
import base64, cv2
from io import BytesIO
import numpy as np
import torch
from torchvision import models
from PIL import Image
    
def ClasObjetos(request):
    return render(request, "Demo88/ClasObjetos.html")

@xframe_options_exempt
def ClasificarObjeto(request):
    #Recibir la imagen desde el Browser o Cliente
    fotoBase64 = request.POST.get("Foto")
    #Convertir la imagen de Base64 a Array de NumPy
    imagenArray = convertirBase64ToNumPy(fotoBase64)
    #Convertir la Imagen de Array a PIL
    imagenPIL = Image.fromarray(imagenArray).convert("RGB")
    #Crear el Modelo usando ResNet y los Pesos de ImageNet    
    pesos = models.ResNet18_Weights.IMAGENET1K_V1
    modelo = models.resnet18(pesos)
    #Procesar la imagen para que tenga la forma del Entrenam.
    preProceso = pesos.transforms()
    imagenProcesada = preProceso(imagenPIL)
    #Aumentar una dimension a la imagen para que sea un array
    entrada = imagenProcesada.unsqueeze(0)
    #Hacer la prediccion
    modelo.eval()
    with torch.no_grad():
        salida = modelo(entrada)
    probabilidades = torch.nn.functional.softmax(salida[0], dim=0)
    top_prob, top_catid = torch.topk(probabilidades, 1)
    with open("imagenet_classes.txt", "r") as file:
        clases = [linea.strip() for linea in file.readlines()]
    print(f"Prediccion: {clases[top_catid]} - Probabilidad: {top_prob.item()}")
    rpta = clases[top_catid] + " - " + str(round(top_prob.item()*100,2)) + "%"
    return HttpResponse(rpta)

def convertirBase64ToNumPy(imagenBase64):
    base64_bytes = imagenBase64.encode('ascii')
    buffer = base64.b64decode(base64_bytes)
    imagenPIL = Image.open(BytesIO(buffer))
    imagen = np.array(imagenPIL)
    return imagen