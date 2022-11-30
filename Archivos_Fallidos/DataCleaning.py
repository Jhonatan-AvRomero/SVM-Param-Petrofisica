# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:52:33 2020

@author: Jhonatan
"""
'''Limpieza de datos'''
print(registro_completo)
plt.figure(figsize=(8,5))
plt.hist(registro_completo['Porosidad'], bins = 10)
plt.show()
plt.boxplot(registro_completo['Porosidad'])
plt.show()
print(registro_completo.describe())

plt.figure(figsize=(8,5))
plt.hist(registro_completo['Saturacion_agua'], bins = 10)

plt.figure(figsize=(8,5))
plt.hist(registro_completo['Permeabilidad'], bins = 10)
plt.show()

columnas = ['Porosidad','Saturacion_agua','Permeabilidad']

Conjunto_Vsh_inicial = registro_completo[columnas]
Conjunto_Vsh_escalado = Escalar_datos(Conjunto_Vsh_inicial)

registro_PCA = Conjunto_Vsh_escalado

from sklearn.decomposition import PCA

pca = PCA(n_components=2, svd_solver='full')
pca.fit(registro_PCA)

resultados_PCA = pca.transform(registro_PCA)
resultados_PCA = pd.DataFrame(resultados_PCA)

def Eliminar_outliers(Conjunto):
    for col in Conjunto.columns:
        col_zscore = str(col) + '_zscore'
        Conjunto[col_zscore] = (Conjunto[col]-Conjunto[col].mean())/Conjunto[col].std(ddof = 0)
        col_outlier = str(col) + '_outlier'
        Conjunto[col_outlier] = (abs(Conjunto[col_zscore]) > 3).astype(int)
        #print(Conjunto.col_outlier.value_counts()[1])
    return(Conjunto)
resultados_PCA = Eliminar_outliers(resultados_PCA)

print(resultados_PCA.describe())
print(resultados_PCA['0_outlier'].value_counts()[1])
print(resultados_PCA['1_outlier'].value_counts()[1])

'''
for col in resultados_PCA.columns:
    col_zscore = str(col) + '_zscore'
    resultados_PCA[col_zscore] = (resultados_PCA[col]-resultados_PCA[col].mean())/resultados_PCA[col].std(ddof = 0)
    resultados_PCA['outlier'] = (abs(resultados_PCA[col_zscore]) > 3).astype(int)
print(resultados_PCA)'''
'''
print(resultados_PCA.outlier.value_counts()[1])

def Eliminar_outliers(Conjunto):
    for col in Conjunto.columns:
        col_zscore = str(col) + '_zscore'
        Conjunto[col_zscore] = (Conjunto[col]-Conjunto[col].mean())/Conjunto[col].std(ddof = 0)
        col_outlier = str(col) + '_outlier'
        Conjunto[col_outlier] = (abs(Conjunto[col_zscore]) > 3).astype(int)
    return(Conjunto)

print(Eliminar_outliers(resultados_PCA))'''

