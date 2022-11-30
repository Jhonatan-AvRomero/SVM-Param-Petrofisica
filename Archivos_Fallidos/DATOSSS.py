# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 22:52:22 2020

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
registro_completo = registro_completo.loc[registro_completo['Pozo'] != 'P5']


registro_completo['Porosidad'] = np.power(registro_completo['Porosidad'],2)

#plt.figure(figsize = (8,8))
#plt.hist(registro_completo['Porosidad'], bins = 10)
#plt.title('Histograma de Porosidad')
#plt.show()

registro_completo['Saturacion_agua'] = np.sqrt(registro_completo['Saturacion_agua'])
#plt.figure(figsize = (8,8))
#plt.hist(registro_completo['Saturacion_agua'], bins = 10)
#plt.title('Histograma de Saturación de agua')
#plt.show()

columnas = ['Porosidad','Saturacion_agua']
registro_pozo = registro_completo[columnas]

#PCA
from sklearn.preprocessing import minmax_scale
Porosidad_escalada = minmax_scale(registro_pozo['Porosidad'])
registro_pozo['Porosidad'] = Porosidad_escalada

#plt.figure(figsize = (10,5))
#plt.scatter(registro_pozo['Saturacion_agua'], registro_pozo['Porosidad'], c = registro_completo['Volumen_arcilla'])
#plt.xlabel('Saturación agua')
#plt.ylabel('Porosidad')
#plt.title('Sw vs Phi')
#plt.show()

#Aplicar Análisis de componentes principales.
#Reducir Porosidad y Saturación de agua.
registro_PCA = registro_pozo
from sklearn.decomposition import PCA

pca = PCA(n_components=1, svd_solver='full')
pca.fit(registro_PCA)

resultados_PCA = pca.transform(registro_PCA)
resultados_PCA = pd.DataFrame(resultados_PCA)
print('\nDescripción: Análisis de componentes principales:')
print(resultados_PCA.describe())

#Rescalar datos del PCA
from sklearn.preprocessing import minmax_scale
PCA = minmax_scale(resultados_PCA)
resultados_PCA = pd.DataFrame(PCA)
resultados_PCA = np.power(resultados_PCA, 2)

#Reescalar Permeabilidad
permeabilidad = registro_completo['Permeabilidad']
from sklearn.preprocessing import minmax_scale
K_ = minmax_scale(permeabilidad)
K_ = pd.DataFrame(K_)
resultados_PCA['K'] = pd.DataFrame(K_)

#Histograma de la componente principal
resultados_PCA[0] = np.sqrt(resultados_PCA[0])
plt.figure(figsize = (8,8))
plt.hist(resultados_PCA[0], bins = 100, density = True)
plt.title('Histograma componentes principales')
plt.show()

plt.figure(figsize=(10,8))
plt.scatter(resultados_PCA[0], resultados_PCA['K'], c = registro_completo['Volumen_arcilla'], s = 10, cmap='jet')
plt.xlabel('PCA')
plt.ylabel('K')
plt.title('PCA vs K')
plt.colorbar()
plt.show()
#__________________________________
from sklearn.preprocessing import minmax_scale
PCA = minmax_scale(resultados_PCA[0])
resultados_PCA[0] = pd.DataFrame(PCA)

plt.figure(figsize = (8,8))
plt.hist(registro_completo['Permeabilidad'])
plt.title('Histograma de Permeabilidad')
plt.show()

registro_K = np.sqrt(resultados_PCA['K'])

resultados_PCA['K'] = pd.DataFrame(registro_K)
resultados_PCA['Profundidad'] = pd.DataFrame(registro_completo['Profundidad'])


plt.figure(figsize=(8,6))
plt.scatter(resultados_PCA[0], resultados_PCA['K'], c = registro_completo['Volumen_arcilla'], s = 10, cmap='jet')
plt.xlabel('PCA')
plt.ylabel('Permeabilidad')
plt.title('Datos escalados')
plt.colorbar()
plt.show()
print(resultados_PCA.describe())

#____________________________________________________________________________
columnas = [0,'K']
X_MODELO = resultados_PCA[columnas]
Y_MODELO = registro_completo['Volumen_arcilla']

from sklearn.model_selection import train_test_split
#Split
X_train, X_test, Y_train, Y_test = train_test_split(X_MODELO, 
                                                    Y_MODELO, test_size=0)
print('X_Train dimensión: ', X_train.shape)
print('X_Test dimensión: ', X_test.shape)
print('Y_Train dimensión: ', Y_train.shape)
print('Y_Test dimensión: ', Y_test.shape)

#Importar librería SVR
from sklearn import svm

#Definir el modelo de SVR
Modelo_SVR = svm.SVR(kernel = 'rbf')
print(Modelo_SVR)

#Entrenar modelo
Modelo_SVR.fit(X_train, Y_train)

#Predecir datos del pozo para evaluar el modelo, se guarda como preddiccion_X_pozo
prediccion_SVR_pozo = Modelo_SVR.predict(X_test)
Y_pred = np.round(prediccion_SVR_pozo,4)

#______________________________Gráfica de los datos de predicción_______________________

plt.figure(figsize=(8,5))
plt.scatter(X_test[0], X_test['K'], c = Y_pred, cmap = 'jet', s = 20)
plt.colorbar()
plt.title('Pozo Predicción')
plt.show()

#Gráfica de los datos reales.
plt.figure(figsize=(8,5))
plt.scatter(X_test[0], X_test['K'], c = Y_test, cmap = 'jet', s = 20)
plt.colorbar()
plt.title('Pozo Real')
plt.show()

#
from sklearn.metrics import mean_squared_error
#Usar: MSE
print(mean_squared_error(Y_test, prediccion_SVR_pozo))
print('Rango de datos reales: ',min(Y_test),max(Y_test))
print('Rango de datos con ML: ',min(Y_pred),max(Y_pred))
#Usar: MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(mean_absolute_percentage_error(Y_test,Y_pred))


#
plt.figure(figsize = (8,8))
plt.scatter(Y_test, Y_pred)
plt.show()