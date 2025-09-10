import torch, cv2
from facenet_pytorch import MTCNN

print("Demo 112: Usar facenet_pytorch para Detectar Rostros en un Video en Tiempo Real")

print("1. Crear un Modelo Detector de Rostros MTCNN")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(margin=40, keep_all=True, factor=0.7, device=device).eval()

print("2. Leer el Video de Disco")
video = cv2.VideoCapture(0, 700)
if(video.isOpened()):
    cuadros = None
    while True:
        rpta, imagen = video.read()
        if(rpta):
            cuadros, ratios = mtcnn.detect(imagen)
            nCaras = 0
            if(cuadros is not None):
                nCaras = len(cuadros)
                for i,cuadro in enumerate(cuadros):
                    x1,y1,x2,y2 = int(cuadro[0]), int(cuadro[1]), int(cuadro[2]), int(cuadro[3])
                    cv2.rectangle(imagen,rec=(x1,y1,x2-x1,y2-y1),color=(0, 255, 0), thickness=5)
                    cv2.putText(imagen, str(round(ratios[i],2)), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
            imagen = cv2.resize(imagen, (800,600))
            cv2.imshow("Video Disco", imagen)
            key = cv2.waitKey(1)
            if(key==ord("s")):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
else:
    print("No se puede leer el Video")