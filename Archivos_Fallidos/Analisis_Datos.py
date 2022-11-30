# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:18:02 2020
@author: Jhonatan
"""
#Cargar las librerías para llevar a cabo los procesos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVR
from sklearn import svm
#Cargar archivo csv = datos_ppp.csv
registro_pozos = pd.read_csv('datos_ppp.csv')
def formato_datos(registro_pozos):
    #Recodificar el nombre de las columnas
    columnas = registro_pozos.columns
    registro_pozos.columns = [str.replace('-','_') for str in columnas]
    ##cols = ['FI', 'FR', 'IK']
    columnas = ['Profundidad', 'Porosidad', 'Permeabilidad', 'Saturacion_agua', 'Volumen_arcilla']
    for columna in columnas:
        registro_pozos[columna] = pd.to_numeric(registro_pozos[columna])
    return registro_pozos
#Todos los datos
registro_pozos = formato_datos(registro_pozos)
print('*'*75, '\nEl tamaño del registro es: ', registro_pozos.shape)
#----------------------------------------------Particionar datos
#Datos Nulos
registro_nulos = registro_pozos.loc[(registro_pozos.columns[2:5] == 0)]
#Datos Completos
registro_comp = registro_pozos.loc[(registro_pozos['Porosidad'] != 0) & (registro_pozos['Permeabilidad'] != 0) & 
                    (registro_pozos['Saturacion_agua'] != 0) & (registro_pozos['Volumen_arcilla'] != 0)]
registro_completos = pd.DataFrame(registro_comp)
registro_completos.sort_values(by=['Profundidad'], inplace = True, ascending = True)
print('El registro completo: ', registro_completos.shape)
#
for columna in registro_pozos.columns[2:6]:
    DF = registro_pozos.loc[(registro_pozos[columna] == 0)]
    if DF.empty == True:
        print('No hay valores nulos para la variable ' + columna)
    else:
        if columna == 'Permeabilidad':
           predecir_permeabilidad = DF
           #print(predecir_permeabilidad)
           predecir_permeabilidad = DF.drop('Permeabilidad', 1)
        print('Sí hay valores nulos en la variable ' + columna, DF.shape)
print('*'*75,'\n',registro_completos.describe().round(4),'\n','*'*75)
#---------------------------Graficar datos
#Histogramas
columnas = registro_completos.columns[1:6]
def grafica_histograma(registro_completos, columnas, bins = 10):
    for columna in columnas:
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        registro_completos[columna].plot.hist(ax = ax, bins = bins)
        ax.set_title('Histograma de ' + columna)
        ax.set_xlabel(columna)
        ax.set_ylabel('Número de datos')
        plt.show()
grafica_histograma(registro_completos, columnas)
#   BOXPLOTS
def graficar_boxplot (registro_completos, columnas, columna_y = 'Permeabilidad'):
    for columna in columnas:
        fig = plt.figure(figsize=(5,5))
        sns.set_style("whitegrid")
        sns.boxplot(columna, data = registro_completos)
        plt.title('Boxplot de ' + columna)
        plt.xlabel(columna)
        plt.ylabel('Permeabilidad')
        plt.show()
cat_cols = ['Porosidad', 'Profundidad', 'Saturacion_agua', 'Volumen_arcilla']
graficar_boxplot(registro_completos, cat_cols)
#Grafica de dispersion
def plot_scatter_size(registro_completos, columnas, columna_y = 'Permeabilidad',
                      columna_tamaño = 'Profundidad', tamaño = 0.00004,
                      alpha = 0.5, MIN  = min(registro_completos['Profundidad'])):
    for columna in columnas:
        temp = registro_completos
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(5,5))
        sns.regplot(columna, columna_y,data = temp, scatter_kws={"alpha":alpha, 
        "s":(registro_completos[columna_tamaño]-MIN)/4}, fit_reg=False)
        plt.show()
plot_scatter_size(registro_completos, columnas)
#PAIR_PLOT
cat_cols = ['Porosidad', 'Permeabilidad', 'Saturacion_agua', 'Volumen_arcilla']
sns.pairplot(registro_completos[cat_cols], diag_kind = "kde")
fig = plt.figure(figsize=(5,5))
plt.show()
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (6,6))
sns.kdeplot(registro_completos['Porosidad'], ax = ax1)
sns.kdeplot(registro_completos['Saturacion_agua'], ax = ax1)
sns.kdeplot(registro_completos['Volumen_arcilla'], ax = ax1)
#print(registro_completos[cat_cols].head(5))
#------------------------Normalizar datos
Matriz_correlación = registro_completos[cat_cols].corr()
print('*'*75,'\nMatriz de correlación:\n\n',Matriz_correlación,'\n','*'*75)
#Normalizar datos completos y guardar (y)
y = registro_completos['Permeabilidad']
y = np.array(y)
registro_completo = registro_completos.drop('Permeabilidad', 1)
df_normalizado = StandardScaler()
registro_completo[['Porosidad', 'Saturacion_agua', 'Volumen_arcilla']] = df_normalizado.fit_transform(registro_completo[['Porosidad', 'Saturacion_agua', 'Volumen_arcilla']])
print('\n\tNormalización de los datos para el modelo\n', registro_completo.describe().round(4),'*'*75)
#
sns.kdeplot(registro_completo['Porosidad'], ax = ax2)
sns.kdeplot(registro_completo['Saturacion_agua'], ax = ax2)
sns.kdeplot(registro_completo['Volumen_arcilla'], ax = ax2)
#
cat_cols = ['Porosidad', 'Saturacion_agua', 'Volumen_arcilla']
sns.pairplot(registro_completo[cat_cols], diag_kind = "kde")
fig = plt.figure(figsize=(5,5))
plt.show()
#Datos_para_el_modelo : -----------------------------Jugar con las variables
Variable1 = registro_completo['Porosidad']
Variable1 = np.array(Variable1)
Variable2 = registro_completo['Volumen_arcilla']
Variable2 = np.array(Variable2)
X = []
i = 0
for v in Variable1:
    X.append([v, Variable2[i]])
    i = i + 1
#Datos_para_predecir
df_normalizado2 = StandardScaler()
predecir_permeabilidad[['Porosidad', 'Saturacion_agua', 'Volumen_arcilla']] = df_normalizado2.fit_transform(predecir_permeabilidad[['Porosidad', 'Saturacion_agua', 'Volumen_arcilla']])
print('*'*75,'\n\tNormalización de los datos para predecir su K\n', predecir_permeabilidad.describe().round(4))
#print(predecir_permeabilidad) ------------------------ Modificar variables
Variable1p = predecir_permeabilidad['Porosidad']
Variable1p = np.array(Variable1p)
Variable2p = predecir_permeabilidad['Volumen_arcilla']
Variable2p = np.array(Variable2p)
print(Variable1p.shape)
print(Variable2p.shape)

Xp = []
i = 0
for v in Variable1p:
    Xp.append([v, Variable2p[i]])
    i = i + 1

#print(predecir_permeabilidad)
#--------------------------------------------
#--------------------------------------------
regr = svm.SVR(kernel = 'rbf')
regr.fit(X, y)
print(regr.fit(X,y))

fig = plt.figure(figsize=(10,10))
plt.scatter(Variable1, Variable2, alpha = 0.05, color = 'blue')
plt.scatter(Variable1p, Variable2p, color = 'red', alpha = 0.09)
plt.plot(X, regr.fit(X,y).predict(X))

y_pred = regr.predict(Xp)
y_column = pd.DataFrame(y_pred)
y_original = pd.DataFrame(y)
print(y_original.describe())
print(y_column.describe())
print('La predicción sugiere: ',y_pred)