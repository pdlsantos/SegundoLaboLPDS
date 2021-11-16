"""
Created on Sat Oct 16 18:25:48 2021

@author: Paulo
"""
import numpy as np
import math
from scipy import interpolate

def rect(x):
    return np.where(np.abs(x)<=0.5, 1 + 0j, 0j)

def ref_chirp(signo, W, tp, fm):
    K = signo*W*(1j)*np.pi/tp
    t = np.linspace(-tp/2, tp/2, int(tp*fm), dtype = np.complex128)
    t2 = t*t
    out = rect(t/tp)*np.exp(K*t2)
    return out

def ref_chirp_DC(signo, W, tp, fm, DC):
    K = signo*W*(1j)*np.pi/tp
    t = np.linspace(-tp/2, tp/2, int(tp*fm), dtype = np.complex128)
    t2 = t*t
    out = rect(t/tp)*np.exp(K*t2)*np.exp(-signo*1j*(2*np.pi*DC*t))
    return out

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def fastconv(A,B):
    lengthC = np.size(A) + np.size(B) - 1
    sizefft = next_power_of_2(lengthC)
    fftA = np.fft.fft(A, n = sizefft, norm = "ortho")
    fftB = np.fft.fft(B, n = sizefft, norm = "ortho")
    fftY = np.sqrt(sizefft)*fftA*fftB
    y = np.fft.ifft(fftY, n = sizefft, norm = "ortho")
    return y[:lengthC]

# SAR
def filtroAdaptadoSAR(y,h):
    shy = np.shape(y)
    output = np.zeros((shy[0], shy[1] + np.size(h)-1), dtype = np.complex128)
    
    for i in range(shy[0]):
        output[i,:] =  np.convolve(h, y[i,:])
    return output 

def filtroAdaptadoAzimut(y,Vr, La, PRF, Fc, signo, Rango):
    shy = np.shape(y)
    output = np.zeros((shy[0], shy[1]), dtype = np.complex128)
    
    for i in range(shy[1]):
        Chirp_Azimut = ref_chirp(signo, 2*Vr*0.886/La, 886*3e8*Rango[i]/(La*Vr*Fc), PRF) # Chirp Down
        h = np.conjugate(Chirp_Azimut)
        h = h[::-1]
        output[:,i] =  np.convolve(h, y[:,i])[int(np.size(Chirp_Azimut)/2):int(np.size(Chirp_Azimut)/2)+shy[0]] 
    
    return output 

def filtroAdaptadoAzimutDC(y,Vr, La, PRF, Fc, signo, Rango,DC):
    shy = np.shape(y)
    output = np.zeros((shy[0], shy[1]), dtype = np.complex128)
    
    for i in range(shy[1]):
        Chirp_Azimut = ref_chirp_DC(signo, 2*Vr*0.886/La, 886*3e8*Rango[i]/(La*Vr*Fc), PRF,DC) # Chirp Down
        h = np.conjugate(Chirp_Azimut)
        h = h[::-1]
        output[:,i] =  np.convolve(h, y[:,i])[int(np.size(Chirp_Azimut)/2):int(np.size(Chirp_Azimut)/2)+shy[0]] 
    
    return output 

def rango(tp,fm, T):
    c = 299792458
    Ro = T*c/2
    Cant = tp*fm-1
    x = np.arange(Cant)
    return (Ro + (x-1)*c/(fm*2))/1000 # En kilometros

def velocidad(PRF, fc, elems):
    Frec = np.linspace(PRF*(-1/2), PRF/2, elems)
    c = 299792458
    lda = c/fc
    return Frec*lda/2

def filtroDopplerSAR(y):
    shy = np.shape(y)
    TL = shy[1]
    TR = shy[0]
    
    output = np.zeros((TR,TL), dtype = np.complex128)
    
    for i in range(TL):
        output[:,i] = np.fft.fft(y[:,i])
    
    return output

def DopplerInversaSAR(y):
    shy = np.shape(y)
    TL = shy[1]
    TR = shy[0]
    z = np.fft.ifftshift(y, axes = 0)
    output = np.zeros((TR,TL), dtype = np.complex128)
    
    for i in range(TL):
        output[:,i] = np.fft.ifft(z[:,i])
    
    return output

def BuscaMaximos(Dato):
    D0 = Dato.shape[0]
    D1 = Dato.shape[1]
    
    output = np.zeros((D0,), dtype = np.complex128)
    for i in range(D0):
        output[i] = np.max(np.abs(Dato[i,:]))
    return output
            

def interpolacion(Datos, Fc, Frec, Vr, Rango):
    output = np.zeros(Datos.shape) ##, dtype = np.complex128
    c = 299792458
    lda = c/Fc
    for k in range(Datos.shape[0]):
        f = interpolate.interp1d(Rango, Datos[k,:], fill_value = "extrapolate")
        R2 = Rango + Rango*0.125* (lda/Vr)**2 * (Frec[k])**2
        output[k,:] = f(R2)
    return output

def CompressionYIFFT(y,Vr, La, PRF, Fc, signo, Rango):
    output = np.zeros((y.shape), dtype = np.complex128)
    c = 299792458
    for i in range(y.shape[1]):
        Chirp_Azimut = ref_chirp(signo, 2*Vr*0.886/La, 886*c*Rango[i]/(La*Vr*Fc), PRF) # Chirp Down
        output[:Chirp_Azimut.size, i] = Chirp_Azimut 
    
    output = np.fft.fft(output, axis = 0)
    return np.fft.ifft(output*y, axis = 0) 

def CompressionYIFFTDC(y,Vr, La, PRF, Fc, signo, Rango, DC):
    output = np.zeros(y.shape, dtype = np.complex128)
    c = 299792458
    for i in range(y.shape[1]):
        Chirp_Azimut = ref_chirp_DC(signo, 2*Vr*0.886/La, 886*c*Rango[i]/(La*Vr*Fc), PRF, DC) # Chirp Down
        output[:Chirp_Azimut.size,i] = Chirp_Azimut 
    
    output = np.fft.fft(output, axis = 0)
    return np.fft.ifft(output*y, axis = 0) 

def interpolacion2(Datos, Fc, Frec, Vr, Rango):
    output = np.copy(Datos) ##, dtype = np.complex128
    c = 299792458
    lda = c/Fc
    for k in range(Datos.shape[0]):
        f = interpolate.interp1d(Rango, Datos[k,:], fill_value = "extrapolate", kind = 'cubic')
        R2 = Rango + Rango*0.125* (lda/Vr)**2 * (Frec[k])**2
        output[k,:] = f(R2)
    return output
