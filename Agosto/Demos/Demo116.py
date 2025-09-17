import torch, cv2
from torch import load
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms import v2 as T
from PIL import Image
import numpy as np

print("Demo 116: Segmentacion de una Imagen de Disco y dibujar personalizado")
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

print("2. Cargar los Pesos del Modelo Pre-Entrenado")
with open('Coco_maskrcnn_0.05.pt', 'rb') as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

print("3. Cargar la imagen de Disco")
archivo = r"C:\Data\Python\2025_06_DADLCV\Imagenes\ImageNet\MasterML.jpg"
imagenPIL = Image.open(archivo).convert("RGB")
imagenArray = np.array(imagenPIL)
imagenArray = cv2.resize(imagenArray, (800,800))
imagenTensor = transform(imagenPIL).unsqueeze(0)
print("imagenTensor: ", imagenTensor.shape)

print("4. Segmentar los objetos en la imagen de entrada")
with torch.no_grad():
    imagen = imagenTensor.view(3, 800, 800).unsqueeze(0).to(device).float()
    print("imagen: ", imagen.shape)
    salida = modelo(imagen)
    print("Salida: ", salida)

print("6. Obtener las mascaras y Ids de Categorias")
imagenMostrar = transformT(imagenPIL)
print("imagenMostrar: ", imagenMostrar.shape)
mascaras = salida[0]["masks"].to("cpu")
cuadros = salida[0]["boxes"].to("cpu")
indicesCat = salida[0]["labels"].to("cpu")
scores = salida[0]["scores"].to("cpu")
nObjetos = len(mascaras)
print("mascaras: ", mascaras.shape)
print("indicesCat: ", indicesCat)
print("scores: ", scores)
print("#objetos: ", nObjetos)

with open("coco.names", "r") as file:
    categorias = [linea.strip() for linea in file.readlines()]
    
print("7. Dibujar las mascaras o segmentos en la imagen")
mascaras = (mascaras>0.5).squeeze(1)
print("mascaras: ", mascaras.shape)
c=0
colores = [(255,0,0), (0,255,0), (0,0,255),(255,255,0), (255,0,255),(0,255,255)]

fig, ejes = plt.subplots()
plt.imshow(imagenArray, cmap="gray")
for i in range(nObjetos):
    if(scores[i]>0.7):
        c=c+1
        x1=cuadros[i][0]
        y1=cuadros[i][1]
        x2=cuadros[i][2]
        y2=cuadros[i][3]
        ancho = x2 - x1
        alto = y2 - y1
        categoria = categorias[indicesCat[i]-1]
        mascara = mascaras[i].cpu().numpy()
        color = colores[i % len(colores)]
        imagenArray[mascara] = color
        #plt.imshow(imagenArray, cmap="gray", alpha=0.2)
        ejes.imshow(imagenArray, cmap="gray", alpha=0.2)
        ejes.text(x1-20,y1-20,categoria,fontsize=20,
            color=tuple(c/255 for c in color),ha='center',va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=3, alpha=0.7))
ejes.set_title("Objetos segmentados: " + str(c))
plt.show()