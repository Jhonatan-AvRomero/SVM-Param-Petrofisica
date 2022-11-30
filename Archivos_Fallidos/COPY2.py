# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 19:30:05 2020

@author: Jhonatan
"""

'''
Conjo = registro_completo[columnas]
print(Conjo.describe())
Conjo = pd.DataFrame(Conjo)
registro_comp = Eliminar_outliers(Conjo)

outliers_porosidad = registro_comp.loc[registro_comp['Porosidad_outlier'] == 1]
outliers_permeabilidad = registro_comp.loc[registro_comp['Permeabilidad_outlier'] == 1]
outliers_saturacion = registro_comp.loc[registro_comp['Saturacion_agua_outlier'] == 1]

print('\n','*'*75,'\nOUTLIERS:')
print(outliers_porosidad.shape)
print(outliers_permeabilidad.shape)
print(outliers_saturacion.shape)
print('\n','*'*75)

#print(registro_comp['Permeabilidad_outlier'].value_counts()[1])
#print(registro_comp['Porosidad_outlier'].value_counts()[1])'''

#Quitar Outliers PCA

registro_con_outlier = resultados_PCA
resultados_sin_outlier = Eliminar_outliers(registro_con_outlier)
Volumen_arcilla = np.array(registro_completo['Volumen_arcilla'])
resultados_sin_outlier['Volumen_arcilla'] = Volumen_arcilla

#Quitar labels de outliers PCA
resultados = resultados_sin_outlier.loc[resultados_sin_outlier['0_outlier'] != 1]
resultados_PC = resultados.loc[resultados['1_outlier'] != 1]

#Quitar outliers de PCA
print(resultados_PC.describe())