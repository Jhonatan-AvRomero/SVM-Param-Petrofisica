# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:44:50 2020

@author: Jhonatan
"""

"""Función Escalar Datos"""
import pandas as pd
from sklearn import preprocessing

def Escalar_datos (Conjunto):
    escalar = preprocessing.StandardScaler().fit(Conjunto)
    Conjunto_escalado = escalar.transform(Conjunto)
    Conjunto_escalado = pd.DataFrame(Conjunto_escalado)