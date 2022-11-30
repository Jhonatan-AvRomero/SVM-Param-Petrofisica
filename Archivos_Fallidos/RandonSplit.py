# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:53:57 2020

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
X_MODELO = registro_completo

from sklearn.model_selection import train_test_split
#Split
X_train, X_test = train_test_split(X_MODELO, test_size=0.5)
print('X_Train dimensión: ', X_train.shape)
print('X_Test dimensión: ', X_test.shape)
print(X_train)