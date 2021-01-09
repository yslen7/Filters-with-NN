import numpy as np
import pyaudio 
import wave
import matplotlib.pyplot as plt
import winsound
from scipy.io import wavfile
import time

plt.close('all')

# Leer la señal
m1, cancion = wavfile.read('senal.wav')
t1 = np.arange(len(cancion)) / float(m1) #Calcular el tiempo de la grabación
cancion = cancion / (2.**15) #Normalización}

#Leer el ruido
m2, ruido = wavfile.read('ruido_lab.wav')
t2 = np.arange(len(ruido)) / float(m2) #Calcular el tiempo de la grabación
ruido = ruido / (2.**15) #Normalización

#Target
target = cancion +  ruido

# Saber si se escucha
#winsound.PlaySound(r'senal.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
# time.sleep(30)

#<========================= Diseñar red neuronal =============================>
alpha_val=[0.1,0.05,0.01,0.005,0.001,0.0001,0.00001]
# Entrenamiento
W=np.random.rand(1,1)
B=np.random.rand(1,1)
WL=np.random.rand(1,1)
for alfa in alpha_val:
    w=W
    b=B
    patron = np.zeros((1,1))
    salida = np.zeros((len(ruido), 1))
    a0=0
    wl=WL
    for i in range(len(ruido)):
        patron = ruido[i]
        a = np.dot(w,patron) + np.dot(wl,a0) + b # Salida de la neurona
        e = target[i] - a        
        salida[i] = e            
        #Actualizar pesos y polarizaciones
        w = w + (2*alfa*e*patron.T)
        b = b + (2*alfa*e)
        wl=wl+(2*alfa*e*a0)
        a0=a
    son_rec = salida * (2.**15) # Sonido recuperado reescalado a su valor original 
    son_rec = np.array(son_rec, dtype = np.int16)
    wavfile.write('filtro.wav', m1, son_rec)
    res=np.corrcoef(cancion.T,salida.T)[0,1]
    #winsound.PlaySound(r'filtro.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
    # time.sleep(30)
    plt.figure(figsize = (10,4))
    plt.plot(t1,salida)
    plt.title('Salida de la neurona con alpha='+str(alfa)+' y res '+str(res))
    plt.xlim(t1[0],t1[-1])
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud de la señal')
    #plt.show()
    
plt.figure(figsize = (10,4))
plt.plot(t1,cancion)
plt.title('Canción original')
plt.xlim(t1[0],t1[-1])
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud de la señal')
plt.figure(figsize = (10,4))
plt.plot(t1,target)
plt.title('Canción + Ruido')
plt.xlim(t1[0],t1[-1])
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud de la señal')
