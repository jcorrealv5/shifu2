from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("Demo 92: Trabajando con el DataSet CelebA")

print("1. Crear el DataSet CelebA")
dst = datasets.CelebA(root="datasets",download=True,
target_type=["bbox","landmarks"],transform=transforms.ToTensor())
print("DataSet: ", dst)

batchSize = 512
print("2. Crear el DataLoader para manejar el DataSet CelebA")
dl = DataLoader(dst, batch_size=batchSize, shuffle=True)
print("DataLoader Test: ", dl)

print("3. Cargar los primeras imagenes del DataLoader")
imagenes, [cuadros,puntos] = next(iter(dl))
print("Cuadros: ", cuadros.shape)
print("Puntos: ", puntos.shape)

print("4. Mostrar la Primera Imagen y su Etiqueta")
imagenTensor, cuadro, punto = imagenes[0], cuadros[0], puntos[0]
x,y,ancho,alto = cuadro[0],cuadro[1],cuadro[2],cuadro[3]

imagenArray = imagenTensor.permute(1, 2, 0)
fig, ejes = plt.subplots()
ejes.imshow(imagenArray)
rect = patches.Rectangle((x, y), ancho, alto,angle=0, fill=True, edgecolor='black', facecolor='blue', linewidth=2)
#ejes.add_patch(rect)
ejes.scatter(punto[::2], punto[1::2], c='red', s=10)
plt.show()