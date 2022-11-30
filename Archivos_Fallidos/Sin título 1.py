# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:04:07 2020

@author: Jhonatan
"""

#Cargar las librerías...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
#________________________________________________________________
registro_completo = pd.read_csv('registro_completo.csv')

plt.figure(figsize = (8,8))
plt.hist(registro_completo['Porosidad'])
plt.title('Histograma Porosidad')
plt.show()
plt.figure(figsize = (8,8))
registro_completo['Porosidad'] = np.power(registro_completo['Porosidad'],3)
plt.title('Histograma Porosidad^2')
plt.hist(registro_completo['Porosidad'], bins = 100)
plt.show()

plt.figure(figsize = (8,8))
plt.hist(registro_completo['Volumen_arcilla'])
plt.title('Histograma Volumen de arcilla')
plt.show()
plt.figure(figsize = (8,8))
registro_completo['Volumen_arcilla'] = np.power(registro_completo['Volumen_arcilla'],3)
plt.title('Histograma Volumen_arcilla')
plt.hist(registro_completo['Volumen_arcilla'], bins = 100)
plt.show()

from sklearn.preprocessing import minmax_scale
PCA = minmax_scale(registro_completo['Porosidad'])
registro_completo['Porosidad'] = PCA

from sklearn.preprocessing import minmax_scale
PCA = minmax_scale(registro_completo['Volumen_arcilla'])
registro_completo['Volumen_arcilla'] = PCA

plt.figure(figsize = (10,8))
plt.scatter(registro_completo['Porosidad'], registro_completo['Volumen_arcilla'], c = registro_completo['Saturacion_agua'], cmap ='jet', s = 9)
plt.show()

