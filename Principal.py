# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 18:20:08 2021

@author: Paulo
"""
import funciones as fun
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pickle

Escribir = True
name = "ImagenSAR"                  # Aca va el nombre del DataCube a leer
sentname = 'DatosTemporales/' + name

    
if Escribir:
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
    print(mat.keys())
    mr_min = 1
    ma_min = 1
    mr_max = mr
    ma_max = ma
    
    ## Chirp de Referencia
     
    Chirp_referencia = fun.ref_chirp(1, chirp_BW, Tp[0], fs[0]) # Chirp Down
    
    
    ## Compresión de pulso en Rango

    s = np.array(Chirp_referencia)
    h = np.conjugate(Chirp_referencia)
    h = h[::-1]
    
    sout = fun.filtroAdaptadoSAR(np.flip(raw, axis = 0),h)
    Paquete = [raw, Fc, La, PRF, Tp, Vr, chirp_BW, fs, ma, ma_min, ma_max, mr, mr_min, mr_max, te, Chirp_referencia, sout]
    with open( sentname, 'wb') as f:
        pickle.dump(Paquete, f)
    print("Datos Guardados en Memoria")
else:
    with open( sentname , 'rb') as f:
        Paquete = pickle.load(f)
    raw, Fc, La, PRF, Tp, Vr, chirp_BW, fs, ma, ma_min, ma_max, mr, mr_min, mr_max, te, Chirp_referencia, sout = Paquete
    print("Datos Recuperados de Memoria")

Rango = fun.rango(mr[0]/fs[0],fs[0], te[0])
c = 299792458
Azimut_max = ma[0]*Vr[0]/PRF[0]

# -----------------------
plt.imshow(10*np.log10(np.abs(raw)), cmap = "gray", aspect = 'auto', extent = (Rango[0], Rango[-1], Azimut_max , 0))
plt.xlabel("Rango [km]")
plt.ylabel("Azimut")
plt.colorbar()
#plt.savefig('Images/Salida_RAW' + ".png")
plt.show()
# -----------------------


# -----------------------
plt.imshow(10*np.log10(np.abs(sout[:,:Rango.size])), cmap = "gray", aspect = 'auto', extent = (Rango[0], Rango[-1], Azimut_max , 0))
plt.xlabel("Rango [km]")
plt.ylabel("Azimut")
plt.colorbar()
#plt.savefig('Images/Salida_Compresión' + ".png")
plt.show()
# -----------------------


# -----------------------
plt.imshow(10*np.log10(np.abs(sout[:,:Rango.size])), cmap = "gray", aspect = 'auto', extent = (Rango[0], Rango[-1], Azimut_max , 0))
plt.xlabel("Rango [km]")
plt.ylabel("Azimut")
plt.xlim(11.55, 11.65)
plt.colorbar()
#plt.savefig('Images/Salida_Compresión_Cerca' + ".png")
plt.show()
# -----------------------

    
sDoppler = fun.filtroDopplerSAR(sout)[:,:Rango.size]
sDopplerShift = np.fft.fftshift(sDoppler, axes = 0)

Frec = np.linspace(PRF[0]*(-1/2), PRF[0]/2, np.size(sDoppler[:,0]))

# -----------------------
plt.imshow(10*np.log10(np.abs(sDopplerShift)), cmap = "gray", aspect = 'auto', extent = (Rango[0], Rango[-1], Frec[0], Frec[-1]))
plt.xlabel("Rango [km]")
plt.ylabel("Frecuencia [Hz]")
plt.xlim(11.55, 11.65)
plt.colorbar()
#plt.savefig('Images/Salida_Doppler' + ".png")
plt.show()
# -----------------------

outputInterpolados = fun.interpolacion(sDopplerShift, Fc[0], Frec, Vr[0], Rango)

# -----------------------
plt.imshow(10*np.log10(np.abs(outputInterpolados)), cmap = "gray", aspect = 'auto', extent = (Rango[0], Rango[-1], Frec[0], Frec[-1]))
plt.xlabel("Rango [km]")
plt.ylabel("Frecuencia [Hz]")
plt.xlim(11.55, 11.65)
plt.colorbar()
#plt.savefig('Images/Salida_Interpolado' + ".png")
plt.show()
# -----------------------


## Descomentar para el blanco puntual
OutputFinalFinal = fun.DopplerInversaSAR(outputInterpolados)
SegundaSalida = fun.filtroAdaptadoAzimut(OutputFinalFinal,Vr[0], La[0], PRF[0], Fc[0], 1, Rango)


## Descomentar para el avion y el satelite
#outputInterpolados = np.fft.ifftshift(outputInterpolados, axes = 0)

## Descomentar para el satelite
# SegundaSalida = fun.CompressionYIFFTDC(outputInterpolados,Vr[0], La[0], PRF[0], Fc[0], 1, Rango, 6900)

## Descomentar para el avion
#SegundaSalida = fun.CompressionYIFFT(outputInterpolados,Vr[0], La[0], PRF[0], Fc[0], 1, Rango)

with open( 'DatosTemporales/' + "ImagenFinalFinal" , 'wb') as f:
    pickle.dump(SegundaSalida, f)
print("Imagen Enviada")

# =============================================================================
# # Para aproximar los maximos finales
# maximos = fun.BuscaMaximos(np.abs(sDopplerShift))
# Min = np.argmax(maximos)
# Max = np.argmax(np.abs(sDopplerShift[Min,:]))
# print(Min, Max)
# R0 = Rango[Max]
# 
# =============================================================================
