import torch, cv2
from torch import load
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import v2 as T
from PIL import Image
import numpy as np
from torchvision.utils import draw_bounding_boxes
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Demo 103: Probar la Deteccion de Objetos")
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((800, 800))
])
transformT = T.Compose([transforms.ToTensor(),T.Resize((800, 800))])

print("1. Crear el Modelo usando un Modelo Predefinido maskrcnn")
modelo = maskrcnn_resnet50_fpn_v2(weights=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = modelo.to(device)
modelo.eval()

print("2. Cargar los Pesos del Modelo Pre-Entrenado")
with open(r"C:\Users\jhonf\Documents\Shifu\Coco_maskrcnn_0.05.pt", 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

print("3. Cargar la imagen de Disco")
archivo = r"C:\Users\jhonf\Documents\Shifu\shifu2\ImageNet\mesa.jpg"
imagenPIL = Image.open(archivo).convert("RGB")
imagenArray = np.array(imagenPIL)
imagenArray = cv2.resize(imagenArray, (800,800))
imagenTensor = transform(imagenPIL).unsqueeze(0)
print("imagenTensor: ", imagenTensor.shape)

print("4. Detectar el objeto en la imagen de entrada")
with torch.no_grad():
    imagen = imagenTensor.view(3, 800, 800).unsqueeze(0).to(device).float()
    print("imagen: ", imagen.shape)
    salida = modelo(imagen)
    print("Salida: ", salida)

print("6. Obtener las salidas con los Cuadros Detectados y Ids de Categorias")
imagenMostrar = transformT(imagenPIL)
print("imagenMostrar: ", imagenMostrar.shape)
cuadros = salida[0]["boxes"].to("cpu")
indicesCat = salida[0]["labels"].to("cpu")
print("cuadros: ", cuadros)
print("indicesCat: ", indicesCat)

print("7. Cargar los Nombres de las Categorias de Coco")
with open("coco.names", "r") as file:
    categorias = [linea.strip() for linea in file.readlines()]

colores=["red", "blue","green","pink","yellow","brown","aqua","orange","purple"]*10
nObjetosDetectados = len(cuadros)
print("nObjetosDetectados: ", nObjetosDetectados)
fig, ejes = plt.subplots()
ejes.imshow(imagenArray, cmap="gray")
c = 0
for i in range(nObjetosDetectados):
    if(indicesCat[i]<80):
        c = c + 1
        x1=cuadros[i][0]
        y1=cuadros[i][1]
        x2=cuadros[i][2]
        y2=cuadros[i][3]
        ancho = x2 - x1
        alto = y2 - y1
        rect = patches.Rectangle((x1, y1), ancho, alto, linewidth=3, edgecolor=colores[i], facecolor='none')
        ejes.add_patch(rect)
        categoria = categorias[indicesCat[i]-1]
        ejes.text(x1-20,y1-20,categoria,fontsize=20,
            color=colores[i],ha='center',va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=3, alpha=0.7))
ejes.set_title("Objetos Detectados: " + str(c))
plt.show()
