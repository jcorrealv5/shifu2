import torch, cv2
import torch.nn as nn
import torch.optim as optim
from torch import load
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


print("Demo 101: Trabajando con el DataSet Coco")
transform = transforms.ToTensor()

print("1. Creando el DataSet CocoDetection")
dsTest = datasets.CocoDetection(root="datasets/coco",annFile="datasets/coco/instances.json")
totalMuestras = len(dsTest)
print("Total de Muestras Test: ", totalMuestras)

print("2. Mostrando la Salida Original")
imagenPIL, salida = dsTest[0]
print("salida antes de Transforms: ", salida)

print("3. Mostrando la Salida Transformada")
dst = datasets.wrap_dataset_for_transforms_v2(dsTest, target_keys=("boxes", "labels"))
imagenPIL, salida = dst[0]
print("salida despues de Transforms: ", salida)

print("4. Crear una imagen con el Dibujo del Cuadro")
imagenTensor = transform(imagenPIL)
print("Shape imagenTensor: ", imagenTensor.shape)
imagenBoxes = draw_bounding_boxes(imagenTensor, salida["boxes"], colors=["red", "blue"], width=3)
imagenArray = imagenBoxes.permute(1,2,0)
print("Shape imagenArray: ", imagenArray.shape)

print("5. Obtener el Nombre de la Categoria del Objeto")
idCategoria = salida["labels"].item()
print("idCategoria: ", idCategoria)
with open("coco.names", "r") as file:
    categorias = [linea.strip() for linea in file.readlines()]
nombreCategoria = categorias[idCategoria-1]

print("5. Dibujar la imagen")
fig, ejes = plt.subplots()
ejes.imshow(imagenArray, cmap="gray")
ejes.set_title(nombreCategoria)
plt.show()