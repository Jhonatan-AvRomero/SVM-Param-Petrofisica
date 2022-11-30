# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:16:26 2020

@author: Jhonatan
"""
#Cargar las librerías...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#1) Cargar archivo csv = datos_ppp.csv
registro_pozos = pd.read_csv('datos_ppp_todos.csv')

#2) Funciones editadas:
def formato_datos(registro_pozos):
    #Recodificar el nombre de las columnas
    columnas = registro_pozos.columns
    registro_pozos.columns = [str.replace('-','_') for str in columnas]
    ##cols = ['FI', 'FR', 'IK']
    columnas = ['Profundidad', 'Porosidad', 'Permeabilidad', 'Saturacion_agua', 'Volumen_arcilla']
    for columna in columnas:
        registro_pozos[columna] = pd.to_numeric(registro_pozos[columna])
    return registro_pozos

#Función de escalado MinMax
def Escalar_datos_minmax (Conjunto):
    from sklearn.preprocessing import minmax_scale
    Conjunto_escalado = minmax_scale(Conjunto)
    Conjunto = Conjunto_escalado
    return(Conjunto)

def Funcion_preproceado_vsh(registro_completo):
    #Elevar al cuadrado la Porosidad
    registro_completo['Porosidad'] = np.power(registro_completo['Porosidad'],2)
    #Raíz cuadrada de Saturacion_agua
    registro_completo['Saturacion_agua'] = np.sqrt(registro_completo['Saturacion_agua'])
    
    #Se usarán estas variables para la predicción
    columnas = ['Porosidad','Saturacion_agua']
    registro_pozo = registro_completo[columnas]
    
    #Escalar la porosidad: Para que esté en el rango de 0 a 1 como Saturacion de agua
    registro_pozo['Porosidad'] = Escalar_datos_minmax(registro_pozo['Porosidad'])
    
    #Aplicar Análisis de componentes principales.
    #Reducir Porosidad y Saturación de agua
    registro_PCA = registro_pozo
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(registro_PCA)
    
    resultados_PCA = pca.transform(registro_PCA)
    resultados_PCA = pd.DataFrame(resultados_PCA)
    print('\nDescripción: Análisis de componentes principales:')
    print(resultados_PCA.describe())
    
    PCA_escalada = Escalar_datos_minmax(resultados_PCA)
    
    registro_K = np.sqrt(registro_completo['Permeabilidad'])
    K_escalada = Escalar_datos_minmax(registro_K)
    
    registro_SVR = pd.DataFrame(PCA_escalada, columns = ['PCA']) 
    registro_SVR['Permeabilidad'] = K_escalada
    print(registro_SVR.describe())
    return(registro_SVR)
#Cargar todos los datos bajo el nombre: registro_pozos
registro_pozos = formato_datos(registro_pozos)
print('*'*75, '\nLa dimensión del registro inicial es: ', registro_pozos.shape, '\n')
#Reemplazar valores 0 por nulos
registro_pozos = registro_pozos.replace({0:np.nan})
print(registro_pozos.info())
'''Hay datos faltantes para Permeabilidad y Volumen_arcilla en los registros.'''

#______________________________________DIVIDIR REGISTROS________________________________________
#TODOS LOS DATOS
#registro_pozos

#DATOS COMPLETOS [TODAS LAS VARIABLES]
registro_completo = registro_pozos.dropna(thresh = 6)
registro_completo = registro_completo.drop_duplicates()
#registro_completo = registro_completo.loc[registro_completo['Saturacion_agua'] != 1]
#registro_completo = registro_completo.loc[registro_completo['Volumen_arcilla'] != 1]
print('\nLa dimensión del registro completo es: ',registro_completo.shape,'\n')
print(registro_completo.info(),'\n','*'*75)
registro_completo.to_csv('protoregistro_completo.csv', index = False)

#DATOS INCOMPLETOS [SIN VOLUMEN ARCILLA]
registro_inc_Vsh = registro_pozos[pd.isnull(registro_pozos['Volumen_arcilla'])]
print('\nLa dimensión del registro [sin arcilla]:',registro_inc_Vsh.shape,'\n')
print(registro_inc_Vsh.info())

#DATOS INCOMPLETOS [SIN PERMEABILIDAD]
registro_inc_K = registro_pozos[pd.isnull(registro_pozos['Permeabilidad'])]
print('\nRegistro sin permeabilidad: ', registro_inc_K.shape,'\n')
print(registro_inc_K.info())
registro_inc_K.to_csv('registro_inc_K.csv', index = False)

#El pozo 5 será para Prueba
#Outliers fuera
#registro_completo = pd.read_csv('registro_completo.csv')
#pozo_test = registro_completo.loc[registro_completo['Pozo'] == 'P5']
#registro_completo = registro_completo.loc[registro_completo['Pozo'] != 'P5']

#Cargar Funciones

registro_SVR = Funcion_preproceado_vsh(registro_completo)
#registro_SVR_test = Funcion_preproceado_vsh(pozo_test)

plt.figure(figsize = (8,8))
plt.scatter(registro_SVR['Permeabilidad'], registro_SVR['PCA'],c = registro_completo['Volumen_arcilla'] ,cmap = 'jet')
plt.xlabel('PCA')
plt.ylabel('Permeabilidad')
plt.title('Análisis de Componentes Principales')
plt.colorbar()
plt.show()  

#________________________

X_MODELO = registro_SVR
Y_MODELO = registro_completo['Volumen_arcilla']

from sklearn.model_selection import train_test_split
#Split
X_train, X_test, Y_train, Y_test = train_test_split(X_MODELO, 
                                                    Y_MODELO, test_size=0.2)#, random_state = 25)
print('X_Train dimensión: ', X_train.shape)
print('X_Test dimensión: ', X_test.shape)
print('Y_Train dimensión: ', Y_train.shape)
print('Y_Test dimensión: ', Y_test.shape)

#Importar librería SVR
from sklearn import svm

#Definir el modelo de SVR
Modelo_SVR = svm.SVR(kernel = 'rbf', C = 0.15, max_iter = 1000)
print(Modelo_SVR)

#Entrenar modelo
Modelo_SVR.fit(X_train, Y_train)

#Predecir datos del pozo para evaluar el modelo, se guarda como preddiccion_X_pozo
#X_test = registro_SVR_test
#Y_test = pozo_test['Volumen_arcilla']
prediccion_SVR_pozo = Modelo_SVR.predict(X_test)
Y_pred = np.round(prediccion_SVR_pozo,4)

#______________________________Gráfica de los datos de predicción_______________________

plt.figure(figsize=(8,5))
plt.scatter(X_test['PCA'], X_test['Permeabilidad'], c = Y_pred, s = 20)
plt.xlabel('PCA')
plt.ylabel('Permeabilidad')
plt.colorbar()
plt.title('Predicción')
plt.show()


#Gráfica de los datos reales.
plt.figure(figsize=(8,5))
plt.scatter(X_test['PCA'], X_test['Permeabilidad'], c = Y_test, s = 20)
plt.xlabel('PCA')
plt.ylabel('Permeabilidad')
plt.colorbar()
plt.title('Real')
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
print('porcentaje de error absoluto medio:' ,mean_absolute_percentage_error(Y_test,Y_pred))

#
plt.figure(figsize = (8,8))
plt.scatter(Y_test, Y_pred)
plt.show()

#Completar el registro de Volumen_arcilla
registro_sin_arcilla = Funcion_preproceado_vsh(registro_inc_Vsh)
prediccion_SVR_pozo = Modelo_SVR.predict(registro_sin_arcilla)
V_faltantes_Vsh = np.round(prediccion_SVR_pozo,4)
plt.figure(figsize=(6,6))
plt.hist(V_faltantes_Vsh)
plt.show()
plt.figure(figsize=(6,6))
plt.hist(registro_completo['Volumen_arcilla'])
plt.show()

#Guardar Volumen_arcilla
registro_inc_Vsh['Volumen_arcilla'] = V_faltantes_Vsh
registro_inc_Vsh.to_csv('Valores_Faltantes_VSH.csv')
print(registro_inc_Vsh.info())

protoregistro_completo = pd.read_csv('protoregistro_completo.csv')
print(protoregistro_completo.info())

protoregistro_completo = protoregistro_completo.append(registro_inc_Vsh)
print(protoregistro_completo.info())

#Guardar archivo final
protoregistro_completo.to_csv('registro_permeabilidad.csv', index = False)