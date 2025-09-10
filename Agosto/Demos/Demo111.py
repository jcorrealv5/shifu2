import torch, cv2
from facenet_pytorch import MTCNN, InceptionResnetV1

print("Demo 111: Usar facenet_pytorch para Detectar y mostrar un rostro en una imagen")

print("1. Crear un Modelo Detector de Rostros MTCNN")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(margin=40, keep_all=True, factor=0.9, device=device).eval()
#print(mtcnn)

print("2. Leer el Archivo de Imagen con las caras")
archivo = r"C:\Users\jhonf\Documents\Shifu\shifu2\ImageNet\black.jpg"
imagen = cv2.imread(archivo)

print("3. Detectar las caras y obtiene las imagenes")
cuadros, ratios = mtcnn.detect(imagen)
nCaras = 0
if(cuadros is not None):
    nCaras = len(cuadros)

    print("4. Mostrar las Caras detectadas con OpenCV")
    for i,cuadro in enumerate(cuadros):
        x1,y1,x2,y2 = int(cuadro[0]), int(cuadro[1]), int(cuadro[2]), int(cuadro[3])
        cv2.rectangle(imagen,rec=(x1,y1,x2-x1,y2-y1),color=(0, 255, 0), thickness=5)
        cv2.putText(imagen, str(round(ratios[i],2)), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
else:
    print("No hay caras detectadas")
imagen = cv2.resize(imagen, (600,600))    
cv2.imshow("Caras Detectadas: " + str(nCaras), imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()