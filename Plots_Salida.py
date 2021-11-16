"""
Test Bench

Created on Tue Oct 26 11:46:16 2021

@author: Paulo De Los Santos

"""
import pickle              # Para Guardar Datos
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import funciones as fun
from scipy import interpolate

with open( 'DatosTemporales/ImagenFinalFinal' , 'rb') as f:
        Imagen = pickle.load(f)
Imagen = 10*np.log10(np.abs(Imagen))

h = 3.9

mat = scipy.io.loadmat('raw_aeropuerto')
    # Datos

Fc = mat['Fc'][0]
La = mat['La'][0]
PRF = mat['PRF'][0]
Tp = mat['Tp'][0]
Vr = mat['Vr'][0]
chirp_BW = mat['chirp_BW'][0]
fs = mat['fs'][0]
ma = mat['ma'][0]
mr = mat['mr'][0]
raw = mat['raw']  ##(2628, 6101) shape
te = mat['te'][0]

Rango = fun.rango(mr[0]/fs[0],fs[0], te[0])
Compensacion = 20*np.log(Rango)
## Para dar vuelta la imagen
Imagen = np.flip(Imagen + Compensacion)

## Para sacarle partes indeseadas
Imagen = Imagen[:,:-100]

## Para que se vea mejor
#plt.imshow(Imagen, cmap = "gray", aspect = 'auto', vmin = 9, vmax = 63, extent=(0, 550, 5.5e4, 0))

plt.imshow(Imagen, cmap = "gray", aspect = 'auto', vmin = 50,  extent = (Rango[0], Rango[-1], 15.36, 0))
plt.xlabel("Rango [km]")
plt.ylabel("Azimut")
#plt.colorbar()
plt.savefig('Images/Salida_ImagenSAR' + ".png")
plt.show()

plt.imshow(Imagen, cmap = "gray", aspect = 'auto', extent = (Rango[0], Rango[-1], 15.36, 0))
plt.xlabel("Rango [km]")
plt.ylabel("Azimut")
plt.xlim(11.55, 11.65)
plt.colorbar()
#plt.savefig('Images/Salida_Puntual2' + ".png")
plt.show()

plt.subplot(3,1,1)
plt.plot(Imagen[:,3079])
plt.title("Corte en Azimut")
plt.xlim(1215, 1415)
plt.subplot(3,1,3)
plt.plot(Imagen[1315,:])
plt.title("Corte en Rango")
plt.xlim(2979, 3179)
plt.ylim(-40, 30)
#plt.savefig('Images/Salida_puntoDeCerca' + ".png")
plt.show()

# 3079
# 1315