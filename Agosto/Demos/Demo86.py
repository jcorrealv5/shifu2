import torch
from torchvision import models

#https://docs.pytorch.org/vision/0.22/models.html
print("Demo 86: Como mostrar una Arquitectura PreDefinida VGGNet")
modelo = models.vgg16(pretrained=False)
print(modelo)

for i,capa in enumerate(modelo.classifier):
    if("out_features" in str(capa)):
        print(f"out_features {i}: {capa.out_features}")
