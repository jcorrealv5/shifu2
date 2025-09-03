import torch
from torch import load
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T

print("Demo 102: Entrenando un Modelo de Deteccion de Objetos usando un Modelo PreEntrenado")
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((800, 800))
])

print("1. Creando el DataSet CocoDetection")
dsTrain = datasets.CocoDetection(root="datasets/coco",annFile="datasets/coco/instances.json",transforms=transform)
totalMuestras = len(dsTrain)
print("Total de Muestras Entrenar: ", totalMuestras)

print("2. Transformando la Salida del DataSet")
dst = datasets.wrap_dataset_for_transforms_v2(dsTrain, target_keys=("boxes", "labels", "masks"))

print("3. Crear el DataLoader")
dlTrain = DataLoader(dst,batch_size=totalMuestras,collate_fn=lambda batch: tuple(zip(*batch)))

print("4. Crear el Modelo usando un Modelo Predefinido maskrcnn")
modelo = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.DEFAULT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = modelo.to(device)
modelo.train()

print("5. Entrenar el Modelo")
optimizer = torch.optim.SGD(modelo.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 200
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(dlTrain):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]        
        loss_dict = modelo(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        #print(f'Epoch [{epoch+1}/{num_epochs}], Item: {c}, Loss: {losses.item():.4f}')
    valorPerdida = losses.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {valorPerdida:.4f}')
torch.save(modelo.state_dict(), "Coco_maskrcnn_" + str(valorPerdida) + ".pt")
