import sys
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread
import os, cv2, time
from PIL import Image
import torch
from torch import nn,load
from torchvision import models, transforms

class Dialogo(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        uic.loadUi("Demo96.ui", self)
        self.txtDirectorio = self.findChild(QtWidgets.QLineEdit, "txtDirectorio")
        btnAbrirDirectorio = self.findChild(QtWidgets.QPushButton, "btnAbrirDirectorio")
        self.cboAtributos = self.findChild(QtWidgets.QComboBox, "cboAtributos")
        btnBuscar = self.findChild(QtWidgets.QPushButton, "btnBuscar")
        self.lblTotal = self.findChild(QtWidgets.QLabel, "lblTotal")
        self.lstArchivos = self.findChild(QtWidgets.QListWidget, "lstArchivos")
        self.lblMensaje = self.findChild(QtWidgets.QLabel, "lblMensaje")
        self.lblFoto = self.findChild(QtWidgets.QLabel, "lblFoto")
        self.chkGuardarFotos = self.findChild(QtWidgets.QCheckBox, "chkGuardarFotos")
        #Llenar el Combo con los 40 atributos desde el archivo
        with open("CelebA_Atributos_Castellano.txt","r") as file:
            self.AtributosNombres = [linea.strip() for linea in file.readlines()]
        self.cboAtributos.addItems(self.AtributosNombres)
        #Programar los eventos de controles
        btnAbrirDirectorio.clicked.connect(self.abrirDirectorio)
        btnBuscar.clicked.connect(self.buscarAtributo)
        #Crear el Modelo ResNet")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        numFeatures = 1024
        dropout=0.2
        clases = 40
        self.modelo.fc = nn.Sequential(nn.Flatten(),
        nn.Linear(self.modelo.fc.in_features, numFeatures),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(numFeatures,clases))
        self.modelo = self.modelo.to(self.device)
        #Cargar el Modelo Pre Entrenado")
        with open('CelebA_ResNet_Attr_4.0.pt', 'rb') as f: 
            self.modelo.load_state_dict(load(f, map_location=self.device, weights_only=True))            
     
    def abrirDirectorio(self):
        directorio = QFileDialog.getExistingDirectory(self, "Selecciona el Directorio con las Fotos", "")
        if(directorio):
            self.lstArchivos.clear()
            self.txtDirectorio.setText(directorio)
            archivos = os.listdir(directorio)
            self.nArchivos = 0
            for archivo in archivos:
                extension = archivo.split(".")[-1]
                if(extension=="jpg" or extension=="png"):
                    self.nArchivos = self.nArchivos + 1
                    self.lstArchivos.addItem(archivo)
            self.lblTotal.setText(str(self.nArchivos))
            
    def buscarAtributo(self):
        pixmap = QPixmap()
        self.lblFoto.setPixmap(pixmap)
        self.lblMensaje.setText("")
        self.atributo = self.cboAtributos.currentIndex()
        self.ruta = self.txtDirectorio.text()
        self.items = []
        for i in range(self.nArchivos):
            item = self.lstArchivos.item(i).text()
            self.items.append(item)
        self.hilo = WorkerAtributos(self)
        self.hilo.encontrado.connect(self.mostrarImagen)
        self.hilo.finalizado.connect(self.mostrarRptaFinal)
        self.hilo.start()
    
    def mostrarImagen(self, item):
        self.lblMensaje.setText(f"Archivo encontrado: {item}")
        archivo = os.path.join(self.ruta,item)
        pixFoto = QPixmap()
        imagen = cv2.imread(archivo)
        ancho, alto = imagen.shape[1], imagen.shape[0]
        qImg = QImage(imagen, ancho, alto, 3 * ancho, QImage.Format_BGR888)
        pixmap = QPixmap(qImg)
        self.lblFoto.setPixmap(pixmap)

    def mostrarRptaFinal(self, mensaje):
        self.lblMensaje.setText(mensaje)
                    
class WorkerAtributos(QThread):
    finalizado = QtCore.pyqtSignal(str)
    encontrado = QtCore.pyqtSignal(str)
    
    def __init__(self, parent):
        super(WorkerAtributos, self).__init__(parent)        
        self.nArchivos = parent.nArchivos
        self.atributo = parent.atributo
        self.ruta = parent.ruta
        self.items = parent.items
        self.modelo = parent.modelo
        self.device = parent.device
    
    def run(self):
        data_transforms = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        c = 0
        for i in range(self.nArchivos):
            item = self.items[i]
            archivo = os.path.join(self.ruta,item)            
            imagenPIL = Image.open(archivo).convert("RGB")
            imagenTensor = data_transforms(imagenPIL).unsqueeze(0)
            self.modelo.eval()
            with torch.no_grad():
                imagenPlana = imagenTensor.view(3, 64, 64).unsqueeze(0).to(self.device)
                salida = self.modelo(imagenPlana)
                rpta = salida[0][self.atributo].item()
                tieneAtributo = (rpta>0.5)
                if(tieneAtributo):
                    c = c + 1
                    self.sleep(1)
                    self.encontrado.emit(item)

        if(c==0):
            self.finalizado.emit("No existe el Atributo buscado")
        else:
            self.finalizado.emit(f"Se encontraron {c} archivos")
                
app = QtWidgets.QApplication(sys.argv)
dlg = Dialogo()
dlg.show()
sys.exit(app.exec_())