from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.clickjacking import xframe_options_exempt
import base64, cv2
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import sys
    
def DeteccionRostros(request):
    return render(request, "Demo114/DeteccionRostros.html")

@xframe_options_exempt
def DetectarRostros(request):
    #Recibir la imagen desde el Browser o Cliente
    fotoBase64 = request.POST.get("Foto")
    #Convertir la imagen de Base64 a Array de NumPy
    imagen = convertirBase64ToNumPy(fotoBase64)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(margin=40, keep_all=True, factor=0.7, device=device).eval()
    cuadros, ratios = mtcnn.detect(imagen)
    if(cuadros is not None):
        for i,cuadro in enumerate(cuadros):
            x1,y1,x2,y2 = int(cuadro[0]), int(cuadro[1]), int(cuadro[2]), int(cuadro[3])
            cv2.rectangle(imagen,rec=(x1,y1,x2-x1,y2-y1),color=(0, 255, 0), thickness=5)
            cv2.putText(imagen, str(round(ratios[i],2)), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
    buffer = convertirNumPyToBytes(imagen)
    return HttpResponse(buffer, content_type="application/octet-stream")

def convertirBase64ToNumPy(imagenBase64):
    base64_bytes = imagenBase64.encode('ascii')
    buffer = base64.b64decode(base64_bytes)
    imagenPIL = Image.open(BytesIO(buffer))
    imagen = np.array(imagenPIL)
    return imagen

def convertirNumPyToBytes(imagen):
    imagenPIL = Image.fromarray(imagen)
    imagenBuffer = BytesIO()
    imagenPIL.save(imagenBuffer, format="PNG")
    rpta = imagenBuffer.getvalue()
    return rpta