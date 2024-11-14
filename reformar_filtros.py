from IPython.display import Image
import numpy as np
import time
import os
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
import glob
import cv2 
import json
from matplotlib import pyplot as plt
import keras



# rutina que imprime imágenes y algunas de sus características
# las imágenes son en escala de grises, flotante
def ver_imgs_gris(noms):
  global imgs
  n=len(noms)
  plt.clf()
  plt.figure(figsize=(6, n*5))
  #plt.figure()#figsize=(20, 4))
  for i in range(n):
      img=imgs[noms[i]]
      print(noms[i],img.shape,'min:',img.min(),'max:',img.max(),'esquina:',img[0,0,0,0])
      #h,w=imgs[i].shape
      ax = plt.subplot(n, 1, i+1)
      plt.imshow(-img[0,:,:,0])
#      plt.imshow(imgs[i].reshape(h, w))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      ax.set_title(noms[i])
  return plt.show()

def reformar_filtros(filtros):
  l=len(filtros)
  m=len(filtros[0])
  n=len(filtros[0][0])
  salida=np.zeros((m,n,1,l))
  print('arreglo',l,n,m,salida.shape)
  for i in range(l):
    for j in range(m):
      for k in range(n):
        #print('ijk',i,j,k,filtros[i][j][k])
        salida[j,k,0,i]=filtros[i][j][k]
  return salida
