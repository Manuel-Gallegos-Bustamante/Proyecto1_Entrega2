#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto1 Entrega2
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
from skimage.filters import threshold_otsu
import nibabel
from scipy.io import loadmat
import os
import glob
import numpy as np
import skimage.io as io
import requests
from sklearn.metrics import jaccard_score
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
monedaURL="https://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/11-Edge-Detection/Hough_Transform_Circles/coins.png"  # se asigna a una variable la url de la imagen que se trabajará en la segunda parte del laboratorio
monedas = requests.get(monedaURL) # se accede a la imagen para su descarga por medio de la url con requests.get
with open("Monedas", "wb") as f: # se trabaja con f como la abreviación para abrir un archivo para escritura "Monedas"
	f.write(monedas.content) #se escribe con .write en el archivo previamente mencionado el contenido de la descarga de la imagen realizado previamente con .content
monedas = io.imread("Monedas") # se carga la imagen del archivo creado con io.imread
vectorColor = monedas.flatten()  #se realiza un .flatten() de la imagen en escala de grises para que se trabaje en una dimensión
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#umbral de binarización de acuerdo al método de Otsu
binOtsu=threshold_otsu(monedas) # calculo del umbral por método Otsu con función threshold_otsu
monedas_binOtsu=monedas>binOtsu # máscasa binaria de la imagen monedas haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
#print(binOtsu) # visualización del umbral calculado con el método Otsu
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#binarización con percentil 60
calculo_percentil60=np.percentile(monedas,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
monedas_percentil60=monedas>calculo_percentil60 # máscasa binaria de la imagen monedas haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
#print(calculo_percentil60) # visualización del umbral calculado con el percentil 60
#binarización con umbral = 175
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
monedas_umbral175 = monedas > 175 # máscasa binaria de la imagen monedas haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#selección de dos umbrales arbitrarios y establecer rango
monedas_copia = monedas.copy() # se crea una copia de la imagen modenas
for i in range(0, len(monedas_copia)): # se realiza un recorrido por las filas de la imagen
	for j in range(0, len(monedas_copia[i])): # se realiza un recorrido por las columnas de la imagen
		if monedas_copia[i][j] > 65 and monedas_copia[i][j] < 250: # Se selecciona un umbral de 65-250 para realizar la binarización de la imagen. Por esta razón se crea una condición en la cual si x pixel (indicado como una posición ij de la matriz de la imagen) es mayor al límite inferior del umbral arbitrario escogido y si es menor al límite superior
			monedas_copia[i][j] = 1 # se cumplirse la condición se asigna al pixel (posición) el color blanco, es decir 1 para que quede la matriz sea binaria
		else: # si el pixel no se encuentra dentro del rango de umbrales establecido
			monedas_copia[i][j] = 0 # se asigna a ese pixel el color negro, es decir 0
#se define función para hacer el cálculo del índice de Jaccard
def Jaccard_index(masck_binaria,anotacion):
	"""
	Función que realiza el cálculo del índice de Jaccard de una márcara binaria y su respectiva anotación que deben tener el mismo tamaño. El índice se calcula con la división de la intersección sobre la unión.
	:param masca_binaria: máscara binaria producto de una segmentación generada por cualquier método
	:param anotacion: anotación de la correspondiente para calcular el índice de Jaccard entre ésta y la máscara binaria ingresada por parámetro
	:return: índice de Jaccard calculado
	"""
	interseccion=0 # se inicializan variables para almacenar los valores de intersecci+on y unión
	union=0
	for i in range(len(masck_binaria)): # recorrido por las filas y columnas de las matrices que entran por parámetro.
		for j in range(len(masck_binaria[0])):
			if masck_binaria[i][j]==1 and anotacion[i][j]==1: # se verifica si la posición evaluada tanto en la máscara como en la anotación es 1
				interseccion+=1 # si se cumple la condición indica que es una intersección y también hace parte de la unión
				union+=1
			elif masck_binaria[i][j]==1 or anotacion[i][j]==1: # si la condición previa no se cumple se verifica si alguna de las posiciones es 1
				union+=1 # de cumplirse la condición indica que es una unión
	indice_Jaccard=interseccion/union # se calcula el índice después de terminados los recorridos dividiendo la intersección entre la unión
	return indice_Jaccard
carga_anotación = loadmat("coins_gt.mat") # se carga archivo de anotaciones de la imagen de monedas con loatmat especifícando commo parámetro el archivo
matriz_anotacion=carga_anotación["gt"] # Se obtiene matriz correspondiente obteniendo la información del atributo "gt"
#pruebas
#visualización de índices calculados y de índice teoría utilizando función propia de Python jaccard_score de la librería sklearn.metrics
print("Otsu")
print(Jaccard_index(monedas_binOtsu,matriz_anotacion))
#print(jaccard_score(monedas_binOtsu.flatten(),matriz_anotacion.flatten()))
print("Percentil60")
print(Jaccard_index(monedas_percentil60,matriz_anotacion))
#print(jaccard_score(monedas_percentil60.flatten(),matriz_anotacion.flatten()))
print("Unbral175")
print(Jaccard_index(monedas_umbral175,matriz_anotacion))
#print(jaccard_score(monedas_umbral175.flatten(),matriz_anotacion.flatten()))
print("Umbral rango")
print(Jaccard_index(monedas_copia,matriz_anotacion))
#print(jaccard_score(monedas_copia.flatten(),matriz_anotacion.flatten()))

#PROBLEMA BIOMÉDICO
archivosresonancias=glob.glob(os.path.join("Heart_Data","Data","*.nii.gz")) #se obtiene una lista de los archivos por medio de glob.glos de la ruta formada por os.path.join como los dos primeros parámetros se indican las carpetas en la cual están los archivos y como tercer parámeto se indica que la lista será de todos los archivos que terminen (tengan el formato) .nii.gz
archivosanotaciones=glob.glob(os.path.join("Heart_Data","GroundTruth","*.nii.gz")) #se obtiene una lista de los archivos por medio de glob.glos de la ruta formada por os.path.join como los dos primeros parámetros se indican las carpetas en la cual están los archivos y como tercer parámeto se indica que la lista será de todos los archivos que terminen (tengan el formato) .nii.gz
info = {} # se crea diccionario vacío para almacenar el nombre de los tres pacientes de las resonancias junto con los valores del número de filas, columnas y cortes que tiene cada una de estas resonancias
for i in archivosresonancias: # se realiza recorrido para todos los archivos de la lista previamente creada. donde i sería la ruta para cada archivo
	carga = nibabel.load(i) # Se carga cada uno de los archivos con nibabel.load
	paciente = (str(carga.header['intent_name']).replace("b'",""))[:-1] # se crea variable que almacenará el nombre del paciente en formato de str. Para esto se accede al atributo'intent_name' con el uso del método .heades. Además se reemplazan caracteres del str que no se desean mantener como lo son b´ con el método .replace el cual recibe como parámetro el str que se desea reemplazar y como 2do parámetro el str por el cual se cambiará. Por último se quita la comilla del final del str con [:-1]
	if paciente not in info: # se verifica que el nombre del paciente no esté en el dict previamente creado
		x, y = carga.shape # accediendo al tamaño que tiene el archivo se asigna a x el número de filas y a y el número de columnas que tiene la resonancia de este paciente
		info[paciente] = {'filas':x, 'columnas':y,'cortes':int(carga.header['slice_end'])} # se una llave para el dict con el nombre del paciente la cual tiene como valor otro dict cuyas llaves corresponden al número de filas y columnas y como última llave el número de cortes al cual se accede con el atributo 'slice_end' con el método header y se convierte a entero
#Se inicializan tres variables de 3 dimensiones las cuales corresponden a las resonancias de cada paciente por lo cual se llama la llave de cada paciente en el dict para cada paciente y el valor la llave que se desea tenga cada dimensión
vol1=np.zeros([info['Patient 12']['filas'], info['Patient 12']['columnas'],info['Patient 12']['cortes']]) # variable para Patient 12
vol2=np.zeros([info['Patient 14']['filas'], info['Patient 14']['columnas'],info['Patient 14']['cortes']]) # variable para Patient 14
vol3=np.zeros([info['Patient 3']['filas'], info['Patient 3']['columnas'],info['Patient 3']['cortes']])# variable para Patient 13
vol1_anotacion=np.zeros([info['Patient 12']['filas'], info['Patient 12']['columnas'],info['Patient 12']['cortes']]) # variable para Patient 12
vol2_anotacion=np.zeros([info['Patient 14']['filas'], info['Patient 14']['columnas'],info['Patient 14']['cortes']]) # variable para Patient 14
vol3_anotacion=np.zeros([info['Patient 3']['filas'], info['Patient 3']['columnas'],info['Patient 3']['cortes']])# variable para Patient 13
#print(vol1.shape, vol2.shape, vol3.shape) # se visualizan dimensiones de variables creadas
for i in range(1,len(archivosresonancias)+1): # se realiza recorrido para todos los archivos de la lista previamente creada. donde i sería la ruta para cada archivo
	carga = nibabel.load(os.path.join("Heart_Data","Data",str(i)+".nii.gz"))# Se carga cada uno de los archivos con nibabel.load
	cargaA = nibabel.load(os.path.join("Heart_Data","GroundTruth",str(i)+".nii.gz"))  # Se carga cada uno de los archivos con nibabel.load
	paciente = (str(carga.header['intent_name']).replace("b'",""))[:-1] # se crea variable que almacenará el nombre del paciente en formato de str. Para esto se accede al atributo'intent_name' con el uso del método .header. Además se reemplazan caracteres del str que no se desean mantener como lo son b´ con el método .replace el cual recibe como parámetro el str que se desea reemplazar y como 2do parámetro el str por el cual se cambiará. Por último se quita la comilla del final del str con [:-1]
	corte = int((str(carga.header['descrip']).replace("b'Slice ", ""))[:-1]) # se crea variable que almacenará el número de corte en formato de int. Para esto se accede al atributo'intent_name' el cual inicialmente con el uso del método .header se toma como str . Además se reemplazan caracteres del str que no se desean mantener como lo son b´Slice  con el método .replace el cual recibe como parámetro el str que se desea reemplazar y como 2do parámetro el str por el cual se cambiará. Por último se quita la comilla del final del str con [:-1]
	if paciente == 'Patient 12':# se crean serie de condicionales para verificar cuál es el nombre del paciente del corte evaluado en el archivo dado por la ruta i
		vol1[:,:,corte] = carga.get_fdata() # En cada uno de los condicionales si se cumple su condición: Se añade la información del corte con el método.get_fdata() al volumen creado para cada paciente, dicha información se asigna para todas las filas y columnas del índice dado por el número del corte
		vol1_anotacion[:, :, corte] = cargaA.get_fdata()
	elif paciente == 'Patient 14':
		vol2[:,:,corte] = carga.get_fdata()
		vol2_anotacion[:, :, corte] = cargaA.get_fdata()
	elif paciente == 'Patient 3':
		vol3[:,:,corte] = carga.get_fdata()
		vol3_anotacion[:, :, corte] = cargaA.get_fdata()
# se definen las variables para almacenar el corte específico de la resonancia y de la anotación respectivamente
vol1corte1,vol1_anotacion1=vol1[:,:,15],vol1_anotacion[:,:,15] # variables para el corte especificado del paciente 12
vol2corte1,vol2_anotacion1=vol2[:,:,15],vol2_anotacion[:,:,15] # variables para el corte especificado del paciente 14
vol3corte1,vol3_anotacion1=vol3[:,:,15],vol3_anotacion[:,:,15] # variables para el corte especificado del paciente 3
vol1corte2,vol1_anotacion2=vol1[:,:,28],vol1_anotacion[:,:,28] # variables para el corte especificado del paciente 12
vol2corte2,vol2_anotacion2=vol2[:,:,28],vol2_anotacion[:,:,28] # variables para el corte especificado del paciente 14
vol3corte2,vol3_anotacion2=vol3[:,:,28],vol3_anotacion[:,:,28] # variables para el corte especificado del paciente 3
vol1corte3,vol1_anotacion3=vol1[:,:,25],vol1_anotacion[:,:,25] # variables para el corte especificado del paciente 12
vol2corte3,vol2_anotacion3=vol2[:,:,25],vol2_anotacion[:,:,25] # variables para el corte especificado del paciente 14
vol3corte3,vol3_anotacion3=vol3[:,:,25],vol3_anotacion[:,:,25] # variables para el corte especificado del paciente 3
#máscaras con binarización Otsu
binOtsu_v1c1=threshold_otsu(vol1corte1) # calculo del umbral por método Otsu con función threshold_otsu
#print(binOtsu_v1c1)
v1c1_binOtsu=vol1corte1>binOtsu_v1c1 # máscasa binaria del corte 1 del volumen1 haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
binOtsu_v2c1=threshold_otsu(vol2corte1) # calculo del umbral por método Otsu con función threshold_otsu
v2c1_binOtsu=vol2corte1>binOtsu_v2c1 # máscasa binaria del corte 1 del volumen2 haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
binOtsu_v3c1=threshold_otsu(vol3corte1) # calculo del umbral por método Otsu con función threshold_otsu
v3c1_binOtsu=vol3corte1>binOtsu_v3c1 # máscasa binaria del corte 1 del volumen3 haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0

binOtsu_v1c2=threshold_otsu(vol1corte2) # calculo del umbral por método Otsu con función threshold_otsu
v1c2_binOtsu=vol1corte2>binOtsu_v1c2 # máscasa binaria del corte 2 del volumen1 haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
binOtsu_v2c2=threshold_otsu(vol2corte2) # calculo del umbral por método Otsu con función threshold_otsu
v2c2_binOtsu=vol2corte2>binOtsu_v2c2 # máscasa binaria del corte 2 del volumen2 haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
binOtsu_v3c2=threshold_otsu(vol3corte2) # calculo del umbral por método Otsu con función threshold_otsu
v3c2_binOtsu=vol3corte2>binOtsu_v3c2 # máscasa binaria del corte 2 del volumen3 haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0

binOtsu_v1c3=threshold_otsu(vol1corte3) # calculo del umbral por método Otsu con función threshold_otsu
v1c3_binOtsu=vol1corte3>binOtsu_v1c3 # máscasa binaria del corte 3 del volumen1 haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
binOtsu_v2c3=threshold_otsu(vol2corte3) # calculo del umbral por método Otsu con función threshold_otsu
v2c3_binOtsu=vol2corte3>binOtsu_v2c3 # máscasa binaria del corte 3 del volumen2 haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
binOtsu_v3c3=threshold_otsu(vol3corte3) # calculo del umbral por método Otsu con función threshold_otsu
v3c3_binOtsu=vol3corte3>binOtsu_v3c3 # máscasa binaria del corte 3 del volumen3 haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
#máscaras con binarización por percentil 60
calculo_percentil60_v1c1=np.percentile(vol1corte1,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
v1c1_percentil60=vol1corte1>calculo_percentil60_v1c1 # máscasa binaria del corte 1 del volumen1  haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
calculo_percentil60_v2c1=np.percentile(vol2corte1,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
v2c1_percentil60=vol2corte1>calculo_percentil60_v2c1 # máscasa binaria del corte 1 del volumen2  haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
calculo_percentil60_v3c1=np.percentile(vol3corte1,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
v3c1_percentil60=vol3corte1>calculo_percentil60_v3c1 # máscasa binaria del corte 1 del volumen3  haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0

calculo_percentil60_v1c2=np.percentile(vol1corte2,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
v1c2_percentil60=vol1corte2>calculo_percentil60_v1c2 # máscasa binaria del corte 2 del volumen1  haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
calculo_percentil60_v2c2=np.percentile(vol2corte2,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
v2c2_percentil60=vol2corte2>calculo_percentil60_v2c2 # máscasa binaria del corte 2 del volumen2  haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
calculo_percentil60_v3c2=np.percentile(vol3corte2,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
v3c2_percentil60=vol3corte2>calculo_percentil60_v3c2 # máscasa binaria del corte 2 del volumen3  haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0

calculo_percentil60_v1c3=np.percentile(vol1corte3,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
v1c3_percentil60=vol1corte3>calculo_percentil60_v1c3 # máscasa binaria del corte 3 del volumen1  haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
calculo_percentil60_v2c3=np.percentile(vol2corte3,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
v2c3_percentil60=vol2corte3>calculo_percentil60_v2c3 # máscasa binaria del corte 3 del volumen2  haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
calculo_percentil60_v3c3=np.percentile(vol3corte3,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
v3c3_percentil60=vol3corte3>calculo_percentil60_v3c3 # máscasa binaria del corte 3 del volumen3  haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
#máscara con umbral 175
v1c1_umbral175 = vol1corte1 > 175 # máscasa binaria del corte 1 del volumen1 haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
v2c1_umbral175 = vol2corte1 > 175 # máscasa binaria del corte 1 del volumen1 haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
v3c1_umbral175 = vol3corte1 > 175 # máscasa binaria del corte 1 del volumen1 haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0

v1c2_umbral175 = vol1corte2 > 175 # máscasa binaria del corte 2 del volumen1 haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
v2c2_umbral175 = vol2corte2 > 175 # máscasa binaria del corte 2 del volumen1 haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
v3c2_umbral175 = vol3corte2 > 175 # máscasa binaria del corte 2 del volumen1 haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0

v1c3_umbral175 = vol1corte3 > 175 # máscasa binaria del corte 3 del volumen1 haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
v2c3_umbral175 = vol2corte3 > 175 # máscasa binaria del corte 3 del volumen1 haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
v3c3_umbral175 = vol3corte3 > 175 # máscasa binaria del corte 3 del volumen1 haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
#máscara con umbral arbitrario
def umbral_65a250(imagen_antes):
	"""
	función para cálculo de binarización por umbral arbitrario de 65-250
	:param imagen_antes: imagen a la cual se le realizará la bianrización
	:return: imagen binarizada
	"""
	imagen=imagen_antes.copy() # copia de la imagen que entra por parámetro para trabajar la binarización
	for i in range(0, len(imagen)): # se realiza un recorrido por las filas de la imagen
		for j in range(0, len(imagen[i])): # se realiza un recorrido por las columnas de la imagen
			if imagen[i][j] > 65 and imagen[i][j] < 250: # Se selecciona un umbral de 65-250 para realizar la binarización de la imagen. Por esta razón se crea una condición en la cual si x pixel (indicado como una posición ij de la matriz de la imagen) es mayor al límite inferior del umbral arbitrario escogido y si es menor al límite superior
				imagen[i][j] = 1 # se cumplirse la condición se asigna al pixel (posición) el color blanco, es decir 1 para que quede la matriz sea binaria
			else: # si el pixel no se encuentra dentro del rango de umbrales establecido
				imagen[i][j] = 0 # se asigna a ese pixel el color negro, es decir 0
	return imagen
v1c1_umbralArb = umbral_65a250(vol1corte1) # máscasa binaria del corte 1 del volumen1 haciendo uso de la función creada para binarización con umbral arbitrario.
v2c1_umbralArb = umbral_65a250(vol2corte1) # máscasa binaria del corte 1 del volumen1 haciendo uso de la función creada para binarización con umbral arbitrario.
v3c1_umbralArb = umbral_65a250(vol3corte1) # máscasa binaria del corte 1 del volumen1 haciendo uso de la función creada para binarización con umbral arbitrario.

v1c2_umbralArb = umbral_65a250(vol1corte2) # máscasa binaria del corte 2 del volumen1 haciendo uso de la función creada para binarización con umbral arbitrario.
v2c2_umbralArb = umbral_65a250(vol2corte2) # máscasa binaria del corte 2 del volumen1 haciendo uso de la función creada para binarización con umbral arbitrario.
v3c2_umbralArb = umbral_65a250(vol3corte2) # máscasa binaria del corte 2 del volumen1 haciendo uso de la función creada para binarización con umbral arbitrario.

v1c3_umbralArb = umbral_65a250(vol1corte3) # máscasa binaria del corte 3 del volumen1 haciendo uso de la función creada para binarización con umbral arbitrario
v2c3_umbralArb = umbral_65a250(vol2corte3) # máscasa binaria del corte 3 del volumen1 haciendo uso de la función creada para binarización con umbral arbitrario.
v3c3_umbralArb = umbral_65a250(vol3corte3) # máscasa binaria del corte 3 del volumen1 haciendo uso de la función creada para binarización con umbral arbitrario.

input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#cálculo índice de Jaccard con función creada previamente, para cada uno de los cortes elegidos de cada volumen. Se indica como 1er parámetro la máscara de binarización para cada método de binarización y, como 2do parámetro, la anotación a la que corresponde al corte y volumen
v1c1_iJaccardOtsu=Jaccard_index(v1c1_binOtsu,vol1_anotacion1) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 por el método Otsu
print(v1c1_iJaccardOtsu) # se imprime el valor del índice
v2c1_iJaccardOtsu=Jaccard_index(v2c1_binOtsu,vol2_anotacion1) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 por el método Otsu
print(v2c1_iJaccardOtsu) # se imprime el valor del índice
v3c1_iJaccardOtsu=Jaccard_index(v3c1_binOtsu,vol3_anotacion1) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 por el método Otsu
print(v3c1_iJaccardOtsu) # se imprime el valor del índice
v1c2_iJaccardOtsu=Jaccard_index(v1c2_binOtsu,vol1_anotacion2) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 por el método Otsu
print(v1c2_iJaccardOtsu) # se imprime el valor del índice
v2c2_iJaccardOtsu=Jaccard_index(v2c2_binOtsu,vol2_anotacion2) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 por el método Otsu
print(v2c2_iJaccardOtsu) # se imprime el valor del índice
v3c2_iJaccardOtsu=Jaccard_index(v3c2_binOtsu,vol3_anotacion2) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 por el método Otsu
print(v3c2_iJaccardOtsu) # se imprime el valor del índice
v1c3_iJaccardOtsu=Jaccard_index(v1c3_binOtsu,vol1_anotacion3) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 por el método Otsu
print(v1c3_iJaccardOtsu) # se imprime el valor del índice
v2c3_iJaccardOtsu=Jaccard_index(v2c3_binOtsu,vol2_anotacion3) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 por el método Otsu
print(v2c3_iJaccardOtsu) # se imprime el valor del índice
v3c3_iJaccardOtsu=Jaccard_index(v3c3_binOtsu,vol3_anotacion3) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 por el método Otsu
print(v3c3_iJaccardOtsu) # se imprime el valor del índice
v1c1_iJaccardP60=Jaccard_index(v1c1_percentil60,vol1_anotacion1) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 por el método del percentil 60
print(v1c1_iJaccardP60) # se imprime el valor del índice
v2c1_iJaccardP60=Jaccard_index(v2c1_percentil60,vol2_anotacion1) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 por el método del percentil 60
print(v2c1_iJaccardP60) # se imprime el valor del índice
v3c1_iJaccardP60=Jaccard_index(v3c1_percentil60,vol3_anotacion1) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 por el método del percentil 60
print(v3c1_iJaccardP60) # se imprime el valor del índice
v1c2_iJaccardP60=Jaccard_index(v1c2_percentil60,vol1_anotacion2) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 por el método del percentil 60
print(v1c2_iJaccardP60) # se imprime el valor del índice
v2c2_iJaccardP60=Jaccard_index(v2c2_percentil60,vol2_anotacion2) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 por el método del percentil 60
print(v2c2_iJaccardP60) # se imprime el valor del índice
v3c2_iJaccardP60=Jaccard_index(v3c2_percentil60,vol3_anotacion2) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 por el método del percentil 60
print(v3c2_iJaccardP60) # se imprime el valor del índice
v1c3_iJaccardP60=Jaccard_index(v1c3_percentil60,vol1_anotacion3) # se crea la variable que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 por el método del percentil 60
print(v1c3_iJaccardP60) # se imprime el valor del índice
v2c3_iJaccardP60=Jaccard_index(v2c3_percentil60,vol2_anotacion3)  # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 por el método del percentil 60
print(v2c3_iJaccardP60) # se imprime el valor del índice
v3c3_iJaccardP60=Jaccard_index(v3c3_percentil60,vol3_anotacion3) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 por el método del percentil 60
print(v3c3_iJaccardP60) # se imprime el valor del índice
v1c1_iJaccardU175=Jaccard_index(v1c1_umbral175,vol1_anotacion1) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 con un umbral de 175
print(v1c1_iJaccardU175) # se imprime el valor del índice
v2c1_iJaccardU175=Jaccard_index(v2c1_umbral175,vol2_anotacion1)  # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 con un umbral de 175
print(v2c1_iJaccardU175) # se imprime el valor del índice
v3c1_iJaccardU175=Jaccard_index(v3c1_umbral175,vol3_anotacion1)  # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 con un umbral de 175
print(v3c1_iJaccardU175) # se imprime el valor del índice
v1c2_iJaccardU175=Jaccard_index(v1c2_umbral175,vol1_anotacion2) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 con un umbral de 175
print(v1c2_iJaccardU175) # se imprime el valor del índice
v2c2_iJaccardU175=Jaccard_index(v2c2_umbral175,vol2_anotacion2) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 con un umbral de 175
print(v2c2_iJaccardU175) # se imprime el valor del índice
v3c2_iJaccardU175=Jaccard_index(v3c2_umbral175,vol3_anotacion2) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 con un umbral de 175
print(v3c2_iJaccardU175) # se imprime el valor del índice
v1c3_iJaccardU175=Jaccard_index(v1c3_umbral175,vol1_anotacion3) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 con un umbral de 175
print(v1c3_iJaccardU175) # se imprime el valor del índice
v2c3_iJaccardU175=Jaccard_index(v2c3_umbral175,vol2_anotacion3) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 con un umbral de 175
print(v2c3_iJaccardU175) # se imprime el valor del índice
v3c3_iJaccardU175=Jaccard_index(v3c3_umbral175,vol3_anotacion3) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 con un umbral de 175
print(v3c3_iJaccardU175) # se imprime el valor del índice
v1c1_iJaccardUArb=Jaccard_index(v1c1_umbralArb,vol1_anotacion1) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 con un umbral arbitrario
print(v1c1_iJaccardUArb)  # se imprime el valor del índice
v2c1_iJaccardUArb=Jaccard_index(v2c1_umbralArb,vol2_anotacion1) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 con un umbral arbitrario
print(v2c1_iJaccardUArb) # se imprime el valor del índice
v3c1_iJaccardUArb=Jaccard_index(v3c1_umbralArb,vol3_anotacion1) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 con un umbral arbitrario
print(v3c1_iJaccardUArb) # se imprime el valor del índice
v1c2_iJaccardUArb=Jaccard_index(v1c2_umbralArb,vol1_anotacion2) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 con un umbral arbitrario
print(v1c2_iJaccardUArb) # se imprime el valor del índice
v2c2_iJaccardUArb=Jaccard_index(v2c2_umbralArb,vol2_anotacion2) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 con un umbral arbitrario
print(v2c2_iJaccardUArb) # se imprime el valor del índice
v3c2_iJaccardUArb=Jaccard_index(v3c2_umbralArb,vol3_anotacion2) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 con un umbral arbitrario
print(v3c2_iJaccardUArb) # se imprime el valor del índice
v1c3_iJaccardUArb=Jaccard_index(v1c3_umbralArb,vol1_anotacion3) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 12 con un umbral arbitrario
print(v1c3_iJaccardUArb) # se imprime el valor del índice
v2c3_iJaccardUArb=Jaccard_index(v2c3_umbralArb,vol2_anotacion3) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 14 con un umbral arbitrario
print(v2c3_iJaccardUArb) # se imprime el valor del índice
v3c3_iJaccardUArb=Jaccard_index(v3c3_umbralArb,vol3_anotacion3) # se crea la varianle que contiene el valor del índice de Jaccard para el corte especificado del paciente 3 con un umbral arbitrario
print(v3c3_iJaccardUArb) # se imprime el valor del índice
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure("Segmentaciones") # se crea la figura para colocar los sublots con las segmentaciones
plt.subplot(6,2,1) # para cada subplot se indican como 1er parámetro el número de filas, como 2do parámetro el número de columnas y como 3er parámetro el índice en el cual irá la o segmentación
plt.title("Vol1 corte1 original") # se visualiza cada una de las máscaras y segmentaciones con imshow y el mapa de color "gray". Además, se le inserta el título con .tittle y se le quitan los ejes con axis("off)
plt.imshow(vol1corte1,cmap="gray")
plt.axis("off")
plt.subplot(6,2,2)
plt.title("Vol3 corte1 original")
plt.imshow(vol3corte1,cmap="gray")
plt.axis("off")
plt.subplot(6,2,3)
plt.title("Otsu vol1 corte1")
plt.imshow(v1c1_binOtsu,cmap="gray")
plt.axis("off")
plt.subplot(6,2,4)
plt.title("Otsu vol3 corte1")
plt.imshow(v3c1_binOtsu,cmap="gray")
plt.axis("off")
plt.subplot(6,2,5)
plt.title("Percentil60 vol1 corte1")
plt.imshow(v1c1_percentil60,cmap="gray")
plt.axis("off")
plt.subplot(6,2,6)
plt.title("Percentil60 vol3 corte1")
plt.imshow(v3c1_percentil60,cmap="gray")
plt.axis("off")
plt.subplot(6,2,7)
plt.title("Umbral175 vol1 corte1")
plt.imshow(v1c1_umbral175,cmap="gray")
plt.axis("off")
plt.subplot(6,2,8)
plt.title("Umbral175 vol3 corte1")
plt.imshow(v3c1_umbral175,cmap="gray")
plt.axis("off")
plt.subplot(6,2,9)
plt.title("Umbral arbitrario vol1 corte1")
plt.imshow(v1c1_umbralArb,cmap="gray")
plt.axis("off")
plt.subplot(6,2,10)
plt.title("Umbral arbitrario vol3 corte1")
plt.imshow(v3c1_umbralArb,cmap="gray")
plt.axis("off")
plt.subplot(6,2,11)
plt.title("Anotación vol1 corte1")
plt.imshow(vol1_anotacion1,cmap="gray")
plt.axis("off")
plt.subplot(6,2,12)
plt.title("Anotación vol3 corte1")
plt.imshow(vol3_anotacion1,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show() # se muestra la figura optimizando el espacio con función tight_layout()
