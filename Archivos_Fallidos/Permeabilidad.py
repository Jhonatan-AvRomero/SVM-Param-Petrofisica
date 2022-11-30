# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:33:00 2020

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

#registro_completo = registro_completo.loc[registro_completo['Pozo'] != 'P5']
#registro_completo = registro_completo.loc[registro_completo['Pozo'] != 'P4']
#registro_completo = registro_completo.loc[registro_completo['Pozo'] != 'P3']
#registro_completo = registro_completo.loc[registro_completo['Pozo'] != 'P2']
#registro_completo[''] = np.log(registro_completo['Saturacion_agua'])
plt.figure(figsize = (8,5))
plt.scatter(registro_completo['Porosidad'], registro_completo['Saturacion_agua'], s = 1)




def Escalar_datos (Conjunto):
    escalar = preprocessing.StandardScaler().fit(Conjunto)
    Conjunto_escalado = escalar.transform(Conjunto)
    Conjunto_escalado = pd.DataFrame(Conjunto_escalado)
    return(Conjunto_escalado)

columnas = ['Porosidad','Saturacion_agua']
#Conjunto_Vsh_inicial = registro_completo[columnas]
#Conjunto_Vsh_escalado = Escalar_datos(Conjunto_Vsh_inicial)
#print(Conjunto_Vsh_escalado.describe())
registro_pozo = registro_completo[columnas]

plt.figure(figsize=(2,10))
plt.plot(registro_pozo['Saturacion_agua'], registro_completo['Profundidad'])

plt.plot(registro_pozo['Porosidad'], registro_completo['Profundidad'])

plt.xlim(0,1)
plt.ylim(min(registro_completo['Profundidad']), max(registro_completo['Profundidad']))

plt.show()

plt.figure(figsize = (10,5))
plt.scatter(registro_pozo['Porosidad'], registro_pozo['Saturacion_agua'], c = registro_completo['Volumen_arcilla'])
plt.xlabel('Porosidad')
plt.ylabel('Saturación agua')
plt.show()
#___________________________Análisis de Componentes Principales__________________
#registro_PCA: Datos de entrada
from sklearn.preprocessing import minmax_scale
PCA = minmax_scale(registro_pozo['Porosidad'])
registro_pozo['Porosidad'] = PCA

plt.figure(figsize = (10,5))
plt.scatter(registro_pozo['Saturacion_agua'], registro_pozo['Porosidad'], c = registro_completo['Volumen_arcilla'])
plt.xlabel('Saturación agua')
plt.ylabel('Porosidad')
plt.show()

registro_PCA = registro_pozo

from sklearn.decomposition import PCA

pca = PCA(n_components=1, svd_solver='full')
pca.fit(registro_PCA)

resultados_PCA = pca.transform(registro_PCA)
resultados_PCA = pd.DataFrame(resultados_PCA)
print(resultados_PCA.describe())

from sklearn.preprocessing import minmax_scale
PCA = minmax_scale(resultados_PCA)
resultados_PCA = pd.DataFrame(PCA)

permeabilidad = registro_completo['Permeabilidad']
from sklearn.preprocessing import minmax_scale
K_ = minmax_scale(permeabilidad)
K_ = pd.DataFrame(K_)

resultados_PCA['K'] = pd.DataFrame(K_)

#Histograma de la componente principal
plt.figure(figsize = (8,8))
plt.hist(resultados_PCA[0], bins = 100)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(resultados_PCA[0], resultados_PCA['K'], c = registro_completo['Volumen_arcilla'], s = 10, cmap='jet')
plt.colorbar()
plt.show()
print(resultados_PCA.describe())
'''
#____________________________________________________________________________

X_MODELO = resultados_PCA
Y_MODELO = registro_completo['Volumen_arcilla']

from sklearn.model_selection import train_test_split
#Split
X_train, X_test, Y_train, Y_test = train_test_split(X_MODELO, 
                                                    Y_MODELO, test_size=0.2)
print('X_Train dimensión: ', X_train.shape)
print('X_Test dimensión: ', X_test.shape)
print('Y_Train dimensión: ', Y_train.shape)
print('Y_Test dimensión: ', Y_test.shape)

#Importar librería SVR
from sklearn import svm

#Definir el modelo de SVR
Modelo_SVR = svm.SVR(kernel = 'rbf', C = 90000)
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
print(mean_absolute_percentage_error(Y_test,Y_pred))'''