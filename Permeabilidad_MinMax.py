# -*- coding: utf-8 -*-
"""
Creat on Thu Sep 10 22:23:30 2020
@author: Jhonatan
"""
#Cargar las librerías...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
#1) Cargar archivo csv = datos_ppp.csv
registro_pozos = pd.read_csv('protoregistro_completo.csv')

def Escalar_datos_minmax (Conjunto):
    from sklearn.preprocessing import minmax_scale
    Conjunto_escalado = minmax_scale(Conjunto)
    Conjunto = Conjunto_escalado
    return(Conjunto)

def Escalar_datos (Conjunto):
    escalar = preprocessing.StandardScaler().fit(Conjunto)
    Conjunto_escalado = escalar.transform(Conjunto)
    Conjunto_escalado = pd.DataFrame(Conjunto_escalado)
    return(Conjunto_escalado)
    
def Funcion_preprocesado_K(registro_completo):
    #Elevar al cuadrado la Porosidad
    registro_completo['Porosidad'] = np.power(registro_completo['Porosidad'],2)
    #Raíz cuadrada de Saturacion_agua
    registro_completo['Volumen_arcilla'] = np.power(registro_completo['Volumen_arcilla'],2)    
    #Se usarán estas variables para la predicción
    columnas = ['Saturacion_agua','Volumen_arcilla']
    registro_pozo = registro_completo[columnas]
    
    registro_PCA = registro_pozo
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(registro_PCA)
    
    resultados_PCA = pca.transform(registro_PCA)
    resultados_PCA = pd.DataFrame(resultados_PCA)
    print('\nDescripción: Análisis de componentes principales:')
    print(resultados_PCA.describe())
    
    PCA_escalada = (resultados_PCA)
    
    registro_SVR = pd.DataFrame(PCA_escalada, columns = ['PCA'])
    registro_SVR['PCA'] = np.power(registro_SVR['PCA'],2)
    registro_SVR['Porosidad'] = np.float64(registro_completo['Porosidad'])
    return(registro_SVR)

PCA = Funcion_preprocesado_K(registro_pozos)
print(PCA.describe())

plt.figure(figsize = (8,8))
plt.scatter(PCA['PCA'], registro_pozos['Porosidad'], c = registro_pozos['Permeabilidad'], cmap = 'jet')
plt.colorbar()
plt.show()

columnas = ['Volumen_arcilla','Saturacion_agua']

X_MODELO = PCA
print(X_MODELO.describe)
Y_MODELO = registro_pozos['Permeabilidad']

from sklearn.model_selection import train_test_split
#Split
X_train, X_test, Y_train, Y_test = train_test_split(X_MODELO, 
                                                    Y_MODELO, test_size=0.5)
print('X_Train dimensión: ', X_train.shape)
print('X_Test dimensión: ', X_test.shape)
print('Y_Train dimensión: ', Y_train.shape)
print('Y_Test dimensión: ', Y_test.shape)

#Importar librería SVR
from sklearn import svm

#Definir el modelo de SVR
Modelo_SVR = svm.SVR(kernel = 'rbf', C = 0.15, gamma = 1000)

#Entrenar modelo
Modelo_SVR.fit(X_train, Y_train)

#Y_test = pozo_test['Volumen_arcilla']1
prediccion_SVR_pozo = Modelo_SVR.predict(X_test)
Y_pred = prediccion_SVR_pozo
Y_pred = pd.DataFrame(Y_pred)

#______________________________Gráfica de los datos de predicción_______________________

plt.figure(figsize=(8,5))
plt.scatter(X_test['PCA'], X_test['Porosidad'], c = Y_pred[0], s = 20, cmap='jet')
plt.colorbar()
plt.title('Pozo Predicción')
plt.show()

#X_test['PCA'], X_test['Porsidad']
#Gráfica de los datos reales.
plt.figure(figsize=(8,5))
plt.scatter(X_test['PCA'], X_test['Porosidad'], c = Y_test, s = 20, cmap='jet')
plt.colorbar()
plt.title('Pozo Real')
plt.show()

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
registro_sin_permeabilidad = Funcion_preprocesado_K(registro_inc_K)
prediccion_SVR_pozo = Modelo_SVR.predict(registro_sin_permeabilidad)
V_faltantes_K = prediccion_SVR_pozo

#Guardar archivo Final
registro_sin_permeabilidad = pd.read_csv('registro_inc_K.csv')
registro_sin_permeabilidad['Permeabilidad'] = V_faltantes_K
print(registro_sin_permeabilidad.describe())

protoregistro_completo = pd.read_csv('registro_permeabilidad.csv')
protoregistro_completo = protoregistro_completo.append(registro_sin_permeabilidad)
protoregistro_completo = protoregistro_completo.sort_values(by=['Pozo', 'Profundidad'])

protoregistro_completo.to_csv('datos_ppp_FINAL.csv', index = False)
plt.hist(V_faltantes_K)
plt.show()
plt.hist(registro_pozos['Permeabilidad'])
plt.show()