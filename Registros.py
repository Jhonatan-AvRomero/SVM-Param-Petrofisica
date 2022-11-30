 # -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:33:06 2020

@author: Jhonatan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

registro_completo = pd.read_csv('datos_ppp_FINAL.csv')

#registro_completo = pd.read_csv('datos_ppp_todos.csv')
registro_inc_k = pd.read_csv('Valores_faltantes_K.csv')

pozos = registro_completo['Pozo'].unique()
columnas = ['Porosidad', 'Saturacion_agua', 'Volumen_arcilla', 'Permeabilidad']
colores = ['purple', 'green', 'red', 'blue']
n = len(columnas)
p = 1
for pozo in pozos:
    nombre_pozo = 'Pozo ' + str(p)
    fig, ax = plt.subplots(1,n, figsize = (8,15), sharey = True)
    fig.suptitle(str(nombre_pozo), fontsize = 16)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, top = 0.95)
    registro_pozo = registro_completo.loc[registro_completo['Pozo'] == pozo]
    registro_k = registro_inc_k.loc[registro_inc_k['Pozo'] == pozo]
    i = 0
    for columna in columnas:
        ax[i].scatter(registro_pozo[columna], registro_pozo['Profundidad'], 
          s = 0, c = colores[i], label = '')
        ax[i].plot(registro_pozo[columna], registro_pozo['Profundidad'], 
          linewidth = 2, c = colores[i], alpha = 0.6, label = '')
        ax[i].set_title(str(columna), fontsize = 10)
        if columna == 'Saturacion_agua':
            ax[i].set_xlim(min(registro_pozo[columna]),1)
        elif columna == 'Permeabilidad':
            ax[i].set_xlim(min(registro_pozo[columna]),0.6)
            ax[i].scatter(registro_k[columna], registro_k['Profundidad'], 
              s = 8, c = 'black', label = 'Valores determinados mediante SVR', 
              marker = 9)            
        else:
            ax[i].set_xlim(min(registro_pozo[columna]),max(registro_pozo[columna]))
        
        ax[i].set_ylim(max(registro_pozo['Profundidad']), 
                 min(registro_pozo['Profundidad']))
        ax[i].grid(alpha = 0.8)
        i = i + 1
    fig.legend(loc = 'lower left', 
               bbox_to_anchor=(0, 0.97, 1, 1), shadow = True, facecolor = '#bdffd6')
    nombre = str(pozo)+'.jpg'
    fig.savefig(nombre, dpi = 400)
    p = p + 1