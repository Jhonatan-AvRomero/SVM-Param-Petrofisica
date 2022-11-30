# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:16:17 2020
@author: Jhonatan
"""
#Cargar las librerías...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

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

def Eliminar_outliers(Conjunto):
    for col in Conjunto.columns:
        col_zscore = str(col) + '_zscore'
        Conjunto[col_zscore] = (Conjunto[col]-Conjunto[col].mean())/Conjunto[col].std(ddof = 0)
        col_outlier = str(col) + '_outlier'
        Conjunto[col_outlier] = (abs(Conjunto[col_zscore]) > 3).astype(int)
        #print(Conjunto.col_outlier.value_counts()[1])
    return(Conjunto)

def Escalar_datos (Conjunto):
    escalar = preprocessing.StandardScaler().fit(Conjunto)
    Conjunto_escalado = escalar.transform(Conjunto)
    Conjunto_escalado = pd.DataFrame(Conjunto_escalado)
    return(Conjunto_escalado)

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
print('\nLa dimensión del registro completo es: ',registro_completo.shape,'\n')
print(registro_completo.info(),'\n','*'*75)
registro_completo.to_csv('registro_completo.csv')

#DATOS INCOMPLETOS [SIN VOLUMEN ARCILLA]
registro_inc_Vsh = registro_pozos[pd.isnull(registro_pozos['Volumen_arcilla'])]
print('\nLa dimensión del registro [sin arcilla]:',registro_inc_Vsh.shape,'\n')
print(registro_inc_Vsh.info())

#DATOS INCOMPLETOS [SIN PERMEABILIDAD]
registro_inc_K = registro_pozos[pd.isnull(registro_pozos['Permeabilidad'])]
print('\nRegistro sin permeabilidad: ', registro_inc_K.shape,'\n')
print(registro_inc_K.info())
registro_inc_K.to_csv('registro_inc_K.csv')

#DATOS COMPLETOS [PARA VOLUMEN ARCILLA]
registro_arcilla = registro_pozos[pd.notnull(registro_pozos['Volumen_arcilla'])]
print('\n','*'*75,'Registro para dimension arcilla: ',registro_arcilla.shape,'\n')
print(registro_arcilla.info())

#_______________________________________________GRAFICAS__________________________________
#PARA MODELO DE ARCILLA
plt.figure(figsize = (8, 5))
plt.scatter(registro_completo['Porosidad'], registro_completo['Saturacion_agua'], c = registro_completo['Permeabilidad'], s = 34, cmap = 'jet')
plt.xlabel(r'$\phi$')
plt.ylabel('Sw')
plt.title('Porosidad vs Saturación de agua [Todos los pozos]')
plt.colorbar(label = 'Permeabilidad')
plt.show()

#_________________________________________Cantidad de datos completos por pozo
Num_Pozos = registro_completo['Pozo'].unique()
Frecuencia = registro_completo['Pozo'].value_counts(sort=True)
print('*'*75,'\n',Frecuencia,'\n','*'*75)
'''Usaré el pozo 5 para COMPARAR RESULTADOS y los demás para el MODELO de ML'''

#___________________________Modelo para el volumen de arcilla____________________
#Seleccionar datos y quitar outliers
columnas = ['Saturacion_agua','Permeabilidad']
sns.pairplot(registro_completo)

#Normalizar datos

Conjunto_Vsh_inicial = registro_completo[columnas]
Conjunto_Vsh_escalado = Escalar_datos(Conjunto_Vsh_inicial)
print(Conjunto_Vsh_escalado.describe())

#___________________________Análisis de Componentes Principales__________________
#registro_PCA: Datos de entrada

registro_PCA = Conjunto_Vsh_escalado


from sklearn.decomposition import PCA

pca = PCA(n_components=2, svd_solver='full')
pca.fit(registro_PCA)

resultados_PCA = pca.transform(registro_PCA)
resultados_PCA = pd.DataFrame(resultados_PCA)

print(resultados_PCA.describe())
print(min(registro_pozos['Profundidad']), max(registro_pozos['Profundidad']))
plt.scatter(resultados_PCA[0],registro_completo['Profundidad'], c = registro_completo['Volumen_arcilla'])

#___________________________________________________________________________________________________________


#profundidad = np.array(registro_completo['Profundidad'])
#resultados_PCA['Profundidad'] = profundidad
Conjunto_Vsh_inicial_SVR = resultados_PCA
Conjunto_Vsh_escalado_SVR = Escalar_datos(Conjunto_Vsh_inicial_SVR)
print(Conjunto_Vsh_escalado_SVR.shape)

print(Conjunto_Vsh_escalado_SVR.shape, registro_completo.shape)


#___________________Dividir los datos: Entrenamiento y Prueba (#Permeabilidad_MODELO)
X_MODELO = Conjunto_Vsh_escalado_SVR
Y_MODELO = registro_completo['Volumen_arcilla']

from sklearn.model_selection import train_test_split
#Split
X_train, X_test, Y_train, Y_test = train_test_split(X_MODELO, 
                                                    Y_MODELO, test_size=0.2, random_state = 9)
print('X_Train dimensión: ', X_train.shape)
print('X_Test dimensión: ', X_test.shape)
print('Y_Train dimensión: ', Y_train.shape)
print('Y_Test dimensión: ', Y_test.shape)

#Importar librería SVR
from sklearn import svm

#Definir el modelo de SVR
Modelo_SVR = svm.SVR(kernel = 'rbf', cache_size = 100)
print(Modelo_SVR)

#Entrenar modelo
Modelo_SVR.fit(X_train, Y_train)

#Predecir datos del pozo para evaluar el modelo, se guarda como preddiccion_X_pozo
prediccion_SVR_pozo = Modelo_SVR.predict(X_test)
Y_pred = np.round(prediccion_SVR_pozo,4)

#______________________________Gráfica de los datos de predicción_______________________
plt.figure(figsize=(8,5))
plt.scatter(X_test[0], X_test[1], c = Y_pred, cmap = 'jet', s = 20)
plt.colorbar()
plt.title('Pozo Predicción')
plt.show()

#Gráfica de los datos reales.
plt.figure(figsize=(8,5))
plt.scatter(X_test[0], X_test[1], c = Y_test, cmap = 'jet', s = 20)
plt.colorbar()
plt.title('Pozo Real')
plt.show()

#
from sklearn.metrics import mean_squared_error
#Usar: MSE
print(mean_squared_error(Y_test, prediccion_SVR_pozo))
print('Rango de datos reales: ',min(Y_test),max(Y_test))
print('Rango de datos con ML: ',min(Y_pred),max(Y_pred))

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(mean_absolute_percentage_error(Y_test,Y_pred))













