import sys
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QCheckBox, QListWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, Qt
import os, cv2, shutil
from PIL import Image
import torch
from torch import nn,load
from torchvision import models, transforms

class Dialogo(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        path_ui = os.path.join(os.path.dirname(__file__), "Demo97.ui")
        uic.loadUi(path_ui, self)
        self.txtDirectorio = self.findChild(QtWidgets.QLineEdit, "txtDirectorio")
        btnAbrirDirectorio = self.findChild(QtWidgets.QPushButton, "btnAbrirDirectorio")
        self.lstAtributos = self.findChild(QtWidgets.QListWidget, "lstAtributos")
        btnBuscar = self.findChild(QtWidgets.QPushButton, "btnBuscar")
        self.lblTotal = self.findChild(QtWidgets.QLabel, "lblTotal")
        self.lstArchivos = self.findChild(QtWidgets.QListWidget, "lstArchivos")
        self.lblMensaje = self.findChild(QtWidgets.QLabel, "lblMensaje")
        self.lblFoto = self.findChild(QtWidgets.QLabel, "lblFoto")
        self.chkGuardarFotos = self.findChild(QtWidgets.QCheckBox, "chkGuardarFotos")
        self.chkSeleccionarTodo = self.findChild(QtWidgets.QCheckBox, "chkSeleccionarTodo")
        self.chkSeleccionarTodo.stateChanged.connect(self.seleccionarTodo)
        #Llenar el Combo con los 40 atributos desde el archivo
        ruta = os.path.join(os.path.dirname(__file__), "CelebA_Atributos_Castellano.txt")
        with open(ruta,"r") as file:
            self.AtributosNombres = [linea.strip() for linea in file.readlines()]
        #Crear una lista con Checks
        for atributo in self.AtributosNombres:
            item = QListWidgetItem(atributo)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.lstAtributos.addItem(item)
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
        rutap = os.path.join(os.path.dirname(__file__), "CelebA_ResNet_Attr_4.0.pt")
        with open(rutap, 'rb') as f: 
            self.modelo.load_state_dict(load(f, map_location=self.device, weights_only=True))            
    
    def seleccionarTodo(self):
        seleccion = self.chkSeleccionarTodo.isChecked()
        for i in range(self.lstAtributos.count()):
            self.lstAtributos.item(i).setCheckState(Qt.Checked if seleccion else Qt.Unchecked)
     
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
        self.atributo = self.lstAtributos.currentIndex()
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
        if (self.chkGuardarFotos.isChecked()):
            carpetaFotos = os.path.basename(self.ruta)
            if(not os.path.isdir(carpetaFotos)):
                os.mkdir(carpetaFotos)
            carpetaAtributo = os.path.join(carpetaFotos, self.lstAtributos.currentText())
            if(not os.path.isdir(carpetaAtributo)):
                os.mkdir(carpetaAtributo)
            archivoOrigen = os.path.join(self.ruta, item)
            archivoDestino = os.path.join(carpetaAtributo, item)
            shutil.copy(archivoOrigen, archivoDestino)

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
        self.lstAtributos = parent.lstAtributos
    
    def run(self):
        data_transforms = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        c = 0
        
        self.attrChked=[]
        for i in range(self.lstAtributos.count()):
            if self.lstAtributos.item(i).checkState() == Qt.Checked:
                self.attrChked.append(i)
        
        nCantAtrr = len(self.attrChked)
        coincidencia=0
        atrr=""
        for i in range(self.nArchivos):
            item = self.items[i]
            archivo = os.path.join(self.ruta,item)            
            imagenPIL = Image.open(archivo).convert("RGB")
            imagenTensor = data_transforms(imagenPIL).unsqueeze(0)
            self.modelo.eval()
            with torch.no_grad():
                imagenPlana = imagenTensor.view(3, 64, 64).unsqueeze(0).to(self.device)
                salida = self.modelo(imagenPlana)
                
                for j in range(nCantAtrr):
                    atrr=self.attrChked[j]
                    rpta = salida[0][atrr].item()
                    tieneAtributo = (rpta>0.5)
                    if(tieneAtributo): 
                        coincidencia=coincidencia+1
                

                if(coincidencia == nCantAtrr):
                    c = c + 1
                    self.sleep(1)
                    self.encontrado.emit(item)
                
                coincidencia=0

        if(c==0):
            self.finalizado.emit("No existe el Atributo buscado")
        else:
            self.finalizado.emit(f"Se encontraron {c} archivos")
                
app = QtWidgets.QApplication(sys.argv)
dlg = Dialogo()
dlg.show()
sys.exit(app.exec_())