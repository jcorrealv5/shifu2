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
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Demo 115: Segmentacion de Instancias con una Imagen de Disco")
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
with open(r"C:\Users\jhonf\Documents\Shifu\Coco_maskrcnn_0.05.pt", "rb") as f: 
     modelo.load_state_dict(load(f, map_location=device, weights_only=True))
     modelo.eval()

print("3. Cargar la imagen de Disco")
archivo = r"C:\Users\jhonf\Documents\Shifu\shifu2\ImageNet\P31-16664.jpg"
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
indicesCat = salida[0]["labels"].to("cpu")
scores = salida[0]["scores"].to("cpu")
nObjetos = str(len(mascaras))
print("mascaras: ", mascaras.shape)
print("indicesCat: ", indicesCat)
print("scores: ", scores)
print("#objetos: ", nObjetos)

print("7. Dibujar las mascaras o segmentos en la imagen")
imagenMostrar = imagenTensor.squeeze(0).to(dtype=torch.uint8)
mascaras = mascaras.squeeze(1) > 0.5
print("mascaras: ", mascaras.shape)
imagenMascaras = draw_segmentation_masks(imagenMostrar, mascaras, colors="yellow")
print("imagenMascaras shape: ", imagenMascaras.shape)
imagenMascaras = imagenMascaras.permute(1,2,0)
print("imagenMascaras nuevo shape: ", imagenMascaras.shape)
plt.imshow(imagenMascaras, cmap="gray")
plt.title("Objetos segmentados: " + nObjetos)
plt.show()
