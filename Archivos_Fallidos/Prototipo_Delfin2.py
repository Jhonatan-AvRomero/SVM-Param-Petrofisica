# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:09:31 2020

@author: Jhonatan
"""
#Cargar las librerías para llevar a cabo los procesos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

#Cargar archivos csv = registro_completo.csv
registro_completo = pd.read_csv('registro_completo.csv')
registro_inc_K = pd.read_csv('registro_inc_K.csv')

plt.figure(figsize = (8, 5))
plt.scatter(registro_completo['Volumen_arcilla'], registro_completo['Porosidad'], c = registro_completo['Saturacion_agua'], s = 34, cmap = 'jet')
plt.xlabel('Vsh')
plt.ylabel(r'$\phi$')
plt.title('Volumen de arcilla vs Porosidad [Datos completos]')
plt.colorbar(label = 'índice de saturación de agua')
plt.show()

#Análisis de componentes principales para los datos del modelo
columnas = ['Volumen_arcilla', 'Porosidad', 'Saturacion_agua']
#Preparar conjuntos
Permeabilidad_BLIND = registro_completo.loc[registro_completo['Pozo'] == 'P5']
K_MODELO = registro_completo[registro_completo.Pozo != 'P5']
#K_MODELO = registro_completo

#
registro_PCA_blind = Permeabilidad_BLIND[columnas]
registro_PCA_blind = pd.DataFrame(preprocessing.scale(registro_PCA_blind))

registro_PCA = K_MODELO[columnas]
registro_PCA = pd.DataFrame(preprocessing.scale(registro_PCA))

#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2, svd_solver='full')
pca.fit(registro_PCA)

PCA_results = pca.transform(registro_PCA)
PCA_results = pd.DataFrame(PCA_results)

PCA_results_blind = pca.transform(registro_PCA_blind)
PCA_results_blind = pd.DataFrame(PCA_results_blind)

print(PCA_results.describe())
plt.figure(figsize = (8, 5))
plt.scatter(PCA_results[1],PCA_results[0], c = K_MODELO['Permeabilidad'], cmap = 'jet')
plt.colorbar()
plt.show()

#Pozo para evaluar el modelo
Y_pozo = Permeabilidad_BLIND['Permeabilidad'].values
X_pozo = PCA_results_blind

#Datos para entrenar el modelo
Y_permeabilidad = K_MODELO['Permeabilidad']
X_permeabilidad = PCA_results


#Escalar los datos de X...
from sklearn import preprocessing
#Escalar los datos del modelo
escalar = preprocessing.StandardScaler().fit(X_permeabilidad)
X_K_escalado = escalar.transform(X_permeabilidad)
X_K_escalado = pd.DataFrame(X_K_escalado)
#Escalar los datos del pozo de evaluación del modelo de ML
escalar = preprocessing.StandardScaler().fit(X_pozo)
X_pozo_escalado = escalar.transform(X_pozo)
X_pozo_escalado = pd.DataFrame(X_pozo_escalado)

#Dividir los datos: Entrenamiento y Prueba (#Permeabilidad_MODELO)
from sklearn.model_selection import train_test_split
#Split
X_train, X_test, y_train, y_test = train_test_split(X_permeabilidad, 
                                                    Y_permeabilidad, test_size=0.0, random_state = 23)
print('X_Train dimensión: ', X_train.shape)
print('X_Test dimensión: ', X_test.shape)
print('Y_Train dimensión: ', y_train.shape)
print('Y_Test dimensión: ', y_test.shape)

#_________________________Sección dedicada al modelo de ML________________________
#Importar librería SVR
from sklearn import svm

#Definir el modelo de SVR
Modelo_SVR = svm.SVR(kernel = 'rbf', C = 0.9)
print(Modelo_SVR)

#Entrenar modelo
Modelo_SVR.fit(X_train, y_train)

#Predecir datos del pozo para evaluar el modelo, se guarda como preddiccion_X_pozo
prediccion_SVR_pozo = Modelo_SVR.predict(X_pozo)
prediccion_SVR_pozo = np.round(prediccion_SVR_pozo,4)

#Gráfica de los datos de predicción.
plt.figure(figsize=(8,5))
plt.scatter(X_pozo[0], X_pozo[1], c = prediccion_SVR_pozo, cmap = 'jet')
plt.colorbar()
plt.title('Pozo Predicción')
plt.show()

#Gráfica de los datos reales.
plt.figure(figsize=(8,5))
plt.scatter(X_pozo[0], X_pozo[1], c = Y_pozo, cmap = 'jet')
plt.colorbar()
plt.title('Pozo Real')
plt.show()

#________________________________Evaluar el modelo______________________________________
from sklearn.metrics import mean_squared_error
#Usar: MSE
print(mean_squared_error(Y_pozo, prediccion_SVR_pozo))
print('Rango de datos reales: ',min(Y_pozo),max(Y_pozo))
print('Rango de datos con ML: ',min(prediccion_SVR_pozo),max(prediccion_SVR_pozo))

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(mean_absolute_percentage_error(Y_pozo,prediccion_SVR_pozo))

plt.plot(Y_pozo,Permeabilidad_BLIND['Profundidad'], label = 'real')
plt.plot(prediccion_SVR_pozo,Permeabilidad_BLIND['Profundidad'], label='predicción')
plt.legend()
plt.show()

