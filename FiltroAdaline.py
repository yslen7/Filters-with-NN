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
# Definir arquitectura de red
alpha_val=[0.1,0.01,0.001,0.0001] 
delay_val=[1,2,3,5,7,9,10]#20,30,50,100]
# Entrenamiento
for delay in delay_val:
    W = np.random.rand(1,delay)
    B=np.random.rand(1,1)
    for alfa in alpha_val:
        w=W
        b=B
        patron = np.zeros((delay,1))
        salida = np.zeros((len(ruido), 1))
        for i in range(len(ruido)):
            if (i == 0):
                patron[0] = ruido[i]
            else:
                dd=i-delay
                ad=int((abs(dd)+dd)/2)
                patron[0:i,0]=ruido[i:ad:-1].T
            a = np.dot(w,patron) + b # Salida de la neurona
            e = target[i] - a            
            salida[i] = e         
            #Actualizar pesos y polarizaciones
            w = w + (2*alfa*e*patron.T)
            b = b + (2*alfa*e)    
        son_rec = salida * (2.**15) # Sonido recuperado reescalado a su valor original 
        son_rec = np.array(son_rec, dtype = np.int16)
        wavfile.write('filtro.wav', m1, son_rec)
        res=np.corrcoef(cancion.T,salida.T)[0,1]
        #winsound.PlaySound(r'filtro.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
        # time.sleep(30)
        plt.figure(figsize = (10,4))
        plt.plot(t1,salida)
        plt.title('Salida de la neurona con alpha='+str(alfa)+', '+str(delay)+' delays y res '+str(res))
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