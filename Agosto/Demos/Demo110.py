import torch, cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


print("Demo 110: Usar facenet_pytorch para Detectar un rostro en una imagen")

print("1. Crear un Modelo Detector de Rostros MTCNN")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(margin=40, keep_all=True, factor=0.2, device=device).eval()
#print(mtcnn)

print("2. Leer el Archivo de Imagen con las caras")
archivo = r"C:\Users\jhonf\Documents\Shifu\shifu2\ImageNet\amigos.jpg"
imagen = cv2.imread(archivo)
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

print("3. Mostrar la imagen en Matplotlib")
plt.imshow(imagen, cmap="gray")
plt.show()

print("4. Detectar las caras")
caras = mtcnn(imagen)

print("5. Mostrar las Caras detectadas en Matplotlib")
c=0
figura, ejes = plt.subplots(1,len(caras))
for cara in caras:
    rostro = ((cara.permute(1, 2, 0) + 1) * 127.5).int()
    ejes[c].imshow(rostro, cmap="gray")
    c=c+1
    print(f"cara: {c}", cara)
plt.show()