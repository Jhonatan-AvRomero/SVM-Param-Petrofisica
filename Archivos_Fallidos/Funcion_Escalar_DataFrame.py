# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:09:08 2020

@author: Jhonatan
"""

'''Normalizar ciertas columnas de un conjunto de datos'''
from sklearn import preprocessing
import pandas as pd

def Normalizar_dataframe(Conjunto, columnas):
    Conjunto_para_escalar = Conjunto[columnas]
    escalar = preprocessing.StandardScaler().fit(Conjunto_para_escalar)
    Conjunto_escalado = escalar.transform(Conjunto_para_escalar)
    Conjunto_escalado = pd.DataFrame(Conjunto_escalado)
    return(Conjunto_escalado)