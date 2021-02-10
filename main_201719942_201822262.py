#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto1 Entrega1
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
from skimage.filters import threshold_otsu
import nibabel
from scipy.io import loadmat
import os
import glob
import numpy as np
import skimage.io as io
import requests
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

image_url="https://estaticos.muyinteresante.es/uploads/images/article/57a2ef2a5cafe82d7b8b4567/elefante_0.jpg" # se asigna a una variable la url de la imagen que se trabajará en la primera parte del laboratorio
r=requests.get(image_url) # se accede a la imagen para su descarga por medio de la url con requests.get
with open("Elefantes", "wb") as f: # se trabaja con f como la abreviación para abrir un archivo para escritura "Elefantes"
	f.write(r.content) #se escribe con .write en el archivo previamente mencionado el contenido de la descarga de la imagen realizado previamente con .content
carga_imagen=io.imread("Elefantes") # se carga la imagen del archivo creado con io.imread
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
# se crea una figura VisualizaciónAnotaciones la cual contiene un supplot de 3x2 para la visualización de las diferentes anotaciones y segmentaciones que indica el enunciado
plt.figure("VisualizaciónAnotaciones")
plt.subplot(3,2,1) # para cada subplot se indican como 1er parámetro el número de filas, como 2do parámetro el número de columnas y como 3er parámetro el índice en el cual irá la imagen
plt.title("Imagen a color") # Para cada una de las imágenes se inserta el título con plt.title, se realiza la visualización de la imágen con plt.imshow y se cancela la visualización de los ejes con plt.axis("off")
plt.imshow(carga_imagen) # imagen original a color
plt.axis("off")
plt.subplot(3,2,3)
plt.title("Anotación Clasificación")
plt.imshow(io.imread("Clasificacion.png")) # se carga la imagen correspondiente a la anotación de clasificación con io.imread
plt.axis("off")
plt.subplot(3,2,4)
plt.title("Anotación Detección")
plt.axis("off")
plt.imshow(io.imread("Deteccion.jpeg"))# se carga la imagen correspondiente a la anotación de detección con io.imread
plt.subplot(3,2,5)
plt.title("Anotación Segmentación \nSemántica")
plt.imshow(io.imread("Seg_Semantica.jpeg")) # se carga la imagen correspondiente a la segmentación semántica con io.imread
plt.axis("off")
plt.subplot(3,2,6)
plt.title("Anotación Segmentación \nde Instancias")
plt.imshow(io.imread("Seg_Instancias.jpeg"))# se carga la imagen correspondiente a la segmentación de instancias con io.imread
plt.axis("off")
plt.tight_layout() #se utiliza plt.tight_layout() para evitar que se sobrepongan títulos y se ajusten las imágenes
plt.show() # visualizar la figura con plt.show
plt.savefig("VisualizaciónAnotaciones",Bbox_inches="tight") # se guarda la figura indicando como 2do parámetro Bbox_inches="tight" para evitar que los bordes blancos sean muy gruesos
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
monedaURL="https://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/11-Edge-Detection/Hough_Transform_Circles/coins.png"  # se asigna a una variable la url de la imagen que se trabajará en la segunda parte del laboratorio
monedas = requests.get(monedaURL) # se accede a la imagen para su descarga por medio de la url con requests.get
with open("Monedas", "wb") as f: # se trabaja con f como la abreviación para abrir un archivo para escritura "Monedas"
	f.write(monedas.content) #se escribe con .write en el archivo previamente mencionado el contenido de la descarga de la imagen realizado previamente con .content
monedas = io.imread("Monedas") # se carga la imagen del archivo creado con io.imread
vectorColor = monedas.flatten()  #se realiza un .flatten() de la imagen en escala de grises para que se trabaje en una dimensión
plt.figure("HistogramaMonedas") # se crea figura "HistogramaMonedas" con un subplot de 1x2 para almacenar la imagen original y su respectivo histograma
plt.subplot(1,2,1) # para cada subplot se indican como 1er parámetro el número de filas, como 2do parámetro el número de columnas y como 3er parámetro el índice en el cual irá la imagen o histograma
plt.imshow(monedas,cmap="gray") # se visualiza la imagen con plt.imshow indicando como segundo parámetro el mapa de color para la visualización el cual en este caso es "gray"
plt.title("Imagen monedas") # tanto para el histograma como para la imagen se inserta el título con plt.title
plt.axis('off') # se quitan los ejes
plt.subplot(1,2,2)
plt.hist(vectorColor,bins=256) # para realizar el histograma se trabaja con la imagen en escala de grises vectorizada previamente para que se trabaje en una dimensión. Además se inidica como parámetro bins=256 para que el histograma tenga más divisiones por cada uno de los intermedios entre 0-255 (negro-blanco)
plt.title('Histograma imagen monedas')
plt.tight_layout() #se utiliza plt.tight_layout() para evitar que se sobrepongan títulos y se ajusten las imágenes
plt.show() # visualizar la figura con plt.show
#plt.savefig("HistogramaMonedas",Bbox_inches="tight")
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#umbral de binarización de acuerdo al método de Otsu
binOtsu=threshold_otsu(monedas) # calculo del umbral por método Otsu con función threshold_otsu
monedas_binOtsu=monedas>binOtsu # máscasa binaria de la imagen monedas haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
#print(binOtsu) # visualización del umbral calculado con el método Otsu
"""
plt.figure("BinOtsu") # figura de la binarización de la imagen con el método Otsu, se inserta título con plt.title y se visualiza la máscara con plt.imshow con el mapa de color "gray" y se quitan los ejes con plt.axis("off") y se visualiza con plt.show
plt.title("Binarización de la imagen con Otsu")
plt.imshow(monedas_binOtsu, cmap="gray")
plt.axis('off')
plt.show()
"""
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#binarización con percentil 60
calculo_percentil60=np.percentile(monedas,60) # se calcula el percentil 60 haciendo uso de la función percentile de la librería numpy la cual recibe como 1er parámetro la imagen a la cual se le calculará el percentil y como 2do parámetro el número del percentil que se desea calcular
monedas_percentil60=monedas>calculo_percentil60 # máscasa binaria de la imagen monedas haciendo uso del umbral calculado previamente. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
#print(calculo_percentil60) # visualización del umbral calculado con el percentil 60
"""
plt.figure("Percentil 60") # figura de la binarización de la imagen con el percentil 60, se inserta título con plt.title y se visualiza la máscara con plt.imshow con el mapa de color "gray" y se quitan los ejes con plt.axis("off") y se visualiza con plt.show
plt.title("Binarización de la imagen con percentil 60")
plt.imshow(monedas_percentil60, cmap="gray")
plt.axis('off')
plt.show()
"""
#binarización con umbral = 175
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
monedas_umbral175 = monedas > 175 # máscasa binaria de la imagen monedas haciendo uso del umbral 175. pixeles con valores mayores al umbral toman el valor de 1, de lo contrario toman el valor de 0
"""
plt.figure("Umbral 175") # figura de la binarización de la imagen con el umbral 175,  se inserta título con plt.title y se visualiza la máscara con plt.imshow con el mapa de color "gray" y se quitan los ejes con plt.axis("off") y se visualiza con plt.show
plt.title("Binarización de la imagen con umbral 175")
plt.imshow(monedas_umbral175, cmap="gray")
plt.axis('off')
plt.show()
"""
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#selección de dos umbrales arbitrarios y establecer rango
monedas_copia = monedas.copy() # se crea una copia de la imagen modenas
for i in range(0, len(monedas_copia)): # se realiza un recorrido por las filas de la imagen
	for j in range(0, len(monedas_copia[i])): # se realiza un recorrido por las columnas de la imagen
		if monedas_copia[i][j] > 65 and monedas_copia[i][j] < 250: # Se selecciona un umbral de 65-250 para realizar la binarización de la imagen. Por esta razón se crea una condición en la cual si x pixel (indicado como una posición ij de la matriz de la imagen) es mayor al límite inferior del umbral arbitrario escogido y si es menor al límite superior
			monedas_copia[i][j] = 255 # se cumplirse la condición se asigna al pixel (posición) el color blanco, es decir 255
		else: # si el pixel no se encuentra dentro del rango de umbrales establecido
			monedas_copia[i][j] = 0 # se asigna a ese pixel el color negro, es decir 0
"""
plt.figure("Umbral arbitrario")# figura de la binarización de la imagen con el rango de umbrales arbitrario,  se inserta título con plt.title y se visualiza la máscara con plt.imshow con el mapa de color "gray" y se quitan los ejes con plt.axis("off") y se visualiza con plt.show
plt.title("Umbral arbitrario") 
plt.imshow(monedas_copia, cmap='gray')
plt.axis('off')
plt.show()
"""
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#subplot para máscaras con segmentaciones en escala de grises
plt.figure("MascarasySegmentaciones") # se crea figua con subplot de 2x4 para las distintas máscaras binarias generadas anteriormente y sus respectivas segmentaciones
plt.subplot(2,4,1) # para cada subplot se indican como 1er parámetro el número de filas, como 2do parámetro el número de columnas y como 3er parámetro el índice en el cual irá la máscara o segmentación
plt.imshow(monedas_binOtsu,cmap="gray") # se visualiza cada una de las máscaras y segmentaciones con imshow y el mapa de color "gray". Además, se le inserta el título con .tittle y se le quitan los ejes con axis("off)
plt.title("Máscara 1:\nOtsu")
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(monedas_percentil60,cmap="gray")
plt.title("Máscara 2:\nPercentil 60")
plt.axis('off')
plt.subplot(2,4,3)
plt.imshow(monedas_umbral175,cmap="gray")
plt.title("Máscara 3: Umbral\narbitrario 175")
plt.axis('off')
plt.subplot(2,4,4)
plt.imshow(monedas_copia,cmap="gray")
plt.title("Máscara 4: Umbral\nrango 65-250")
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(monedas_binOtsu*monedas,cmap="gray") # para realizar las segmentaciones se realiza una multiplicación (elemento por elemento) de la imagen original de monedas y de la máscara correspondiente a la segmentación deseada. Lo anterior debido a que como la máscara es binaria (valores de 0 o 1) aquellos pixeles (elementos) que en la máscara tengan un valor de 1 mantendran su nivel de gris mientras que los que tengan un valor de 0 en la máscara binaria tomarán un valor de 0 es decir negro
plt.title("Segmentación 1:\nOtsu")
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(monedas_percentil60*monedas,cmap="gray")
plt.title("Segmentación 2:\nPercentil 60")
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(monedas*monedas_umbral175,cmap="gray")
plt.title("Segmentación 3:\nUmbral 175")
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(monedas_copia * monedas,cmap="gray")
plt.title("Segmentación 4: Umbral\nrango 65-250")
plt.axis('off')
plt.tight_layout() #se utiliza plt.tight_layout() para evitar que se sobrepongan títulos y se ajusten las imágenes
plt.show() # visualizar la figura con plt.show
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#PROBLEMA BIOMÉDICO
archivosresonancias=glob.glob(os.path.join("Heart_Data","Data","*.nii.gz")) #se obtiene una lista de los archivos por medio de glob.glos de la ruta formada por os.path.join como los dos primeros parámetros se indican las carpetas en la cual están los archivos y como tercer parámeto se indica que la lista será de todos los archivos que terminen (tengan el formato) .nii.gz
info = {} # se crea diccionario vacío para almacenar el nombre de los tres pacientes de las resonancias junto con los valores del número de filas, columnas y cortes que tiene cada una de estas resonancias
for i in archivosresonancias: # se realiza recorrido para todos los archivos de la lista previamente creada. donde i sería la ruta para cada archivo
	carga = nibabel.load(i) # Se carga cada uno de los archivos con nibabel.load
	paciente = (str(carga.header['intent_name']).replace("b'",""))[:-1] # se crea variable que almacenará el nombre del paciente en formato de str. Para esto se accede al atributo'intent_name' con el uso del método .heades. Además se reemplazan caracteres del str que no se desean mantener como lo son b´ con el método .replace el cual recibe como parámetro el str que se desea reemplazar y como 2do parámetro el str por el cual se cambiará. Por último se quita la comilla del final del str con [:-1]
	if paciente not in info: # se verifica que el nombre del paciente no esté en el dict previamente creado
		x, y = carga.shape # accediendo al tamaño que tiene el archivo se asigna a x el número de filas y a y el número de columnas que tiene la resonancia de este paciente
		info[paciente] = {'filas':x, 'columnas':y,'cortes':int(carga.header['slice_end'])} # se una llave para el dict con el nombre del paciente la cual tiene como valor otro dict cuyas llaves corresponden al número de filas y columnas y como última llave el número de cortes al cual se accede con el atributo 'slice_end' con el método header y se convierte a entero
	#print(carga) #se visualizan los atributos del archivo
	#Atributo identificar paciente -> intent_name     : b'Patient 3'
	#Atributo identificar #total cortes -> slice_end       : 35
	#Atributo identificar #corte -> descrip         : b'Slice 1'
	#Atributo resolución corte->  dim             : [  2 512 512   1   1   1   1   1]
#Se inicializan tres variables de 3 dimensiones las cuales corresponden a las resonancias de cada paciente por lo cual se llama la llave de cada paciente en el dict para cada paciente y el valor la llave que se desea tenga cada dimensión
vol1=np.zeros([info['Patient 12']['filas'], info['Patient 12']['columnas'],info['Patient 12']['cortes']]) # variable para Patient 12
vol2=np.zeros([info['Patient 14']['filas'], info['Patient 14']['columnas'],info['Patient 14']['cortes']]) # variable para Patient 14
vol3=np.zeros([info['Patient 3']['filas'], info['Patient 3']['columnas'],info['Patient 3']['cortes']])# variable para Patient 13
#print(vol1.shape, vol2.shape, vol3.shape) # se visualizan dimensiones de variables creadas
for i in archivosresonancias: # se realiza recorrido para todos los archivos de la lista previamente creada. donde i sería la ruta para cada archivo
	carga = nibabel.load(i)# Se carga cada uno de los archivos con nibabel.load
	paciente = (str(carga.header['intent_name']).replace("b'",""))[:-1] # se crea variable que almacenará el nombre del paciente en formato de str. Para esto se accede al atributo'intent_name' con el uso del método .header. Además se reemplazan caracteres del str que no se desean mantener como lo son b´ con el método .replace el cual recibe como parámetro el str que se desea reemplazar y como 2do parámetro el str por el cual se cambiará. Por último se quita la comilla del final del str con [:-1]
	corte = int((str(carga.header['descrip']).replace("b'Slice ", ""))[:-1]) # se crea variable que almacenará el número de corte en formato de int. Para esto se accede al atributo'intent_name' el cual inicialmente con el uso del método .header se toma como str . Además se reemplazan caracteres del str que no se desean mantener como lo son b´Slice  con el método .replace el cual recibe como parámetro el str que se desea reemplazar y como 2do parámetro el str por el cual se cambiará. Por último se quita la comilla del final del str con [:-1]
	if paciente == 'Patient 12':# se crean serie de condicionales para verificar cuál es el nombre del paciente del corte evaluado en el archivo dado por la ruta i
		vol1[:,:,corte] = carga.get_fdata() # En cada uno de los condicionales si se cumple su condición: Se añade la información del corte con el método.get_fdata() al volumen creado para cada paciente, dicha información se asigna para todas las filas y columnas del índice dado por el número del corte
	elif paciente == 'Patient 14':
		vol2[:,:,corte] = carga.get_fdata()
	elif paciente == 'Patient 3':
		vol3[:,:,corte] = carga.get_fdata()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#visualización de resonancia para paciente 12 (vol1) en los diferentes ejes cada una de estas visualizaciones se realiza activando el modo interactivo con plt.ion, plt.show y generando un recorrido sobre la longitud del eje que se desea mostrar
plt.ion()
plt.show()
for i in range(len(vol1[0,0])): # recorrido para visualización eje z
	plt.imshow(vol1[:,:,i], cmap='gray') # se visualiza la imagen con el colormap "gray" desactivando los ejes (axis("off")) y "dibujando" sobre la figura el corte de cada iteración junto con su respectivo título
	plt.axis('off')
	plt.title(f'Resonancia paciente 12, eje z, corte {i}')
	plt.draw()
	plt.pause(0.001) # se realiza una pausa de 0.001 segundos para porteriormente cerrar la visualización del corte
	plt.clf()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.ion()
plt.show()
for i in range(len(vol1)): # recorrido para visualización eje x
	plt.imshow(vol1[i,:,:], cmap='gray')# se visualiza la imagen con el colormap "gray" desactivando los ejes (axis("off")) y "dibujando" sobre la figura el corte de cada iteración junto con su respectivo título
	plt.axis('off')
	plt.title(f'Resonancia paciente 12, eje x, corte {i}')
	plt.draw()
	plt.pause(0.001) # se realiza una pausa de 0.001 segundos para porteriormente cerrar la visualización del corte
	plt.clf()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.ion()
plt.show()
for i in range(len(vol1[0])):# recorrido para visualización eje y
	plt.imshow(vol1[:,i,:], cmap='gray') # se visualiza la imagen con el colormap "gray" desactivando los ejes (axis("off")) y "dibujando" sobre la figura el corte de cada iteración junto con su respectivo título
	plt.axis('off')
	plt.title(f'Resonancia paciente 12, eje y, corte {i}')
	plt.draw()
	plt.pause(0.001) # se realiza una pausa de 0.001 segundos para porteriormente cerrar la visualización del corte
	plt.clf()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#visualización de resonancia para paciente 14 (vol2) en los diferentes ejes. cada una de estas visualizaciones se realiza activando el modo interactivo con plt.ion, plt.show y generando un recorrido sobre la longitud del eje que se desea mostrar
plt.ion()
plt.show()
for i in range(len(vol2[0,0])):# recorrido para visualización eje z
	plt.imshow(vol2[:,:,i], cmap='gray')# se visualiza la imagen con el colormap "gray" desactivando los ejes (axis("off")) y "dibujando" sobre la figura el corte de cada iteración junto con su respectivo título
	plt.axis('off')
	plt.title(f'Resonancia paciente 14, eje z, corte {i}')
	plt.draw()
	plt.pause(0.001) # se realiza una pausa de 0.001 segundos para porteriormente cerrar la visualización del corte
	plt.clf()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.ion()
plt.show()
for i in range(len(vol2)):  # recorrido para visualización eje x
	plt.imshow(vol2[i, :, :],cmap='gray')  # se visualiza la imagen con el colormap "gray" desactivando los ejes (axis("off")) y "dibujando" sobre la figura el corte de cada iteración junto con su respectivo título
	plt.axis('off')
	plt.title(f'Resonancia paciente 14, eje x, corte {i}')
	plt.draw()
	plt.pause(0.001)  # se realiza una pausa de 0.001 segundos para porteriormente cerrar la visualización del corte
	plt.clf()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.ion()
plt.show()
for i in range(len(vol2[0])):  # recorrido para visualización eje y
	plt.imshow(vol2[:, i, :],cmap='gray')  # se visualiza la imagen con el colormap "gray" desactivando los ejes (axis("off")) y "dibujando" sobre la figura el corte de cada iteración junto con su respectivo título
	plt.axis('off')
	plt.title(f'Resonancia paciente 14, eje y, corte {i}')
	plt.draw()
	plt.pause(0.001)  # se realiza una pausa de 0.001 segundos para porteriormente cerrar la visualización del corte
	plt.clf()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#visualización de resonancia para paciente 3 (vol3) en los diferentes ejes. cada una de estas visualizaciones se realiza activando el modo interactivo con plt.ion, plt.show y generando un recorrido sobre la longitud del eje que se desea mostrar
plt.ion()
plt.show()
for i in range(len(vol3[0,0])):# recorrido para visualización eje z
	plt.imshow(vol3[:,:,i], cmap='gray') # se visualiza la imagen con el colormap "gray" desactivando los ejes (axis("off")) y "dibujando" sobre la figura el corte de cada iteración junto con su respectivo título
	plt.axis('off')
	plt.title(f'Resonancia paciente 3, eje z, corte {i}')
	plt.draw()
	plt.pause(0.001) # se realiza una pausa de 0.001 segundos para porteriormente cerrar la visualización del corte
	plt.clf()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.ion()
plt.show()
for i in range(len(vol3)): # recorrido para visualización eje x
	plt.imshow(vol3[i,:,:], cmap='gray')# se visualiza la imagen con el colormap "gray" desactivando los ejes (axis("off")) y "dibujando" sobre la figura el corte de cada iteración junto con su respectivo título
	plt.axis('off')
	plt.title(f'Resonancia paciente 3, eje x, corte {i}')
	plt.draw()
	plt.pause(0.001) # se realiza una pausa de 0.001 segundos para porteriormente cerrar la visualización del corte
	plt.clf()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.ion()
plt.show()
for i in range(len(vol3[0])):# recorrido para visualización eje y
	plt.imshow(vol3[:,i,:], cmap='gray') # se visualiza la imagen con el colormap "gray" desactivando los ejes (axis("off")) y "dibujando" sobre la figura el corte de cada iteración junto con su respectivo título
	plt.axis('off')
	plt.title(f'Resonancia paciente 3, eje y, corte {i}')
	plt.draw()
	plt.pause(0.001) # se realiza una pausa de 0.001 segundos para porteriormente cerrar la visualización del corte
	plt.clf()

##

dato = " "
dato.