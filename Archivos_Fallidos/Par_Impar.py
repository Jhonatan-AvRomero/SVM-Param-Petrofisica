# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:22:58 2020

@author: Jhonatan
"""
import pandas as pd
import numpy as np
'''Crear archivos Par y None'''
#Cargar archivo csv = datos_ppp.csv
Nombre_Pozo = 'OUTC98280319'
registro = pd.read_csv('datos_fff_1.csv')

Indicador = np.array([])
i = 0
for dato in registro['DEPTH']:
    if i%2 == 0:
        Indicador = np.append(Indicador, Nombre_Pozo + 'PAR')
    elif i%2 != 0:
        Indicador = np.append(Indicador, Nombre_Pozo + 'IMPAR')
    i = i + 1
registro['Well Name'] = Indicador
print(registro.info())

columnas = ['Well Name','DEPTH','Shale','Dolomite','Limestone','Primary Porosity','Secondary Porosity','Facie C']
registro_final = registro[columnas]
registro_final.to_csv('datos_fff_pi.csv')
'''
#Guardar DataFrames
datos_fff_par = registro.loc[registro['Par'] == 'Par']
datos_fff_par = datos_fff_par.drop(['Par'], axis = 1)
datos_fff_par.to_csv('datos_fff_par.csv')


datos_fff_impar = registro.loc[registro['Par'] == 'Impar']
datos_fff_impar = datos_fff_impar.drop(['Par'], axis = 1)
datos_fff_impar.to_csv('datos_fff_impar.csv')'''
