# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:01:47 2020

@author: Jhonatan
"""

def Escalar_datos (Conjunto):
    from sklearn.preprocessing import minmax_scale
    Conjunto_escalado = minmax_scale(Conjunto)
    Conjunto = Conjunto_escalado
    return(Conjunto)
