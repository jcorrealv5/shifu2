import torch
from torchvision import models

print("Demo 85: Como mostrar una Arquitectura PreDefinida AlexNet")
modelo = models.alexnet(pretrained=False)
print(modelo)

ultimaCapa = modelo.classifier[6]
print("Entrada Ultima Capa: ", ultimaCapa.in_features)
print("Salida Ultima Capa: ", ultimaCapa.out_features)

ultimaCapa.out_features = 10
print("Salida Ultima Capa Modificada: ", ultimaCapa.out_features)