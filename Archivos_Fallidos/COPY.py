# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:32:23 2020

@author: Jhonatan
"""

#__________________________Separación de datos para el modelo___________________
#Datos del pozo que se usarán para evaluar el modelo de ML
#1Y_pozo = Arcilla_BLIND['Volumen_arcilla'].values
#1X_pozo = Arcilla_BLIND.drop(['Pozo','Profundidad','Permeabilidad','Volumen_arcilla'], axis = 1)

#Datos de los pozos para entrenar el modelo de ML
Y_arcilla = Arcilla_MODELO['Volumen_arcilla'].values
X_arcilla = Arcilla_MODELO.drop(['Pozo','Profundidad','Permeabilidad','Volumen_arcilla'], axis = 1)

#Escalar los datos de X...
from sklearn import preprocessing

def Escalar_datos (Conjunto):
    escalar = preprocessing.StandardScaler().fit(Conjunto)
    Conjunto_escalado = escalar.transform(Conjunto)
    Conjunto_escalado = pd.DataFrame(Conjunto_escalado)
    return(Conjunto_escalado)
#Escalar los datos del modelo
X_arcilla_escalado = Escalar_datos(X_arcilla)

#Escalar los datos del pozo de evaluación del modelo de ML
#1escalar = preprocessing.StandardScaler().fit(X_pozo)
#1X_pozo_escalado = escalar.transform(X_pozo)
#1X_pozo_escalado = pd.DataFrame(X_pozo_escalado)
#Datos escalados del modelo...
#1print('*'*75,'\nInforme de Datos escalados',X_arcilla_escalado.describe(), '\n',X_pozo_escalado.describe(),'\n','*'*75)

#Pairplots: Para verificar que han sido escalados los datos.
sns.pairplot(X_arcilla)
plt.show()
sns.pairplot(X_arcilla_escalado)
plt.show()

#Dividir los datos: Entrenamiento y Prueba (#Arcilla_MODELO)
from sklearn.model_selection import train_test_split
#Split
X_train, X_test, y_train, y_test = train_test_split(X_arcilla_escalado, 
                                                    Y_arcilla, test_size=0.2, random_state = 3)
print('X_Train dimensión: ', X_train.shape)
print('X_Test dimensión: ', X_test.shape)
print('Y_Train dimensión: ', y_train.shape)
print('Y_Test dimensión: ', y_test.shape)

#_________________________Sección dedicada al modelo de ML________________________
#Importar librería SVR
from sklearn import svm

#Definir el modelo de SVR
Modelo_SVR = svm.SVR(kernel = 'rbf')
print(Modelo_SVR)

#Entrenar modelo
Modelo_SVR.fit(X_train, y_train)

#Ver bloc Comentarios: Si se usa Test...

#Predecir datos del pozo para evaluar el modelo, se guarda como preddiccion_X_pozo
prediccion_SVR_pozo = Modelo_SVR.predict(X_test)
prediccion_SVR_pozo = np.round(prediccion_SVR_pozo,4)

#Gráfica de los datos de predicción.
plt.figure(figsize=(8,5))
plt.scatter(X_test[0], X_test[1], c = prediccion_SVR_pozo, cmap = 'jet')
plt.colorbar()
plt.title('Pozo Predicción')
plt.show()

#Gráfica de los datos reales.
plt.figure(figsize=(8,5))
plt.scatter(X_test[0], X_test[1], c = y_test, cmap = 'jet')
plt.colorbar()
plt.title('Pozo Real')
plt.show()

#________________________________Evaluar el modelo______________________________________
from sklearn.metrics import mean_squared_error
#Usar: MSE
print(mean_squared_error(y_test, prediccion_SVR_pozo))
print('Rango de datos reales: ',min(y_test),max(y_test))
print('Rango de datos con ML: ',min(prediccion_SVR_pozo),max(prediccion_SVR_pozo))

#Usar MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(mean_absolute_percentage_error(y_test,prediccion_SVR_pozo))

"""
#Grafica de comparación de resultados:
fig, ax = plt.subplots(1, 2, figsize = (6,12), sharey = True)

Vol_arcilla = np.vstack((Arcilla_BLIND['Volumen_arcilla']))
ax[0].imshow(Vol_arcilla, aspect = 'auto', cmap = 'jet', extent=[0,1,max(Arcilla_BLIND['Profundidad']), min(Arcilla_BLIND['Profundidad'])])
ax[0].set_title('Vsh Original')

Vol_arcilla_SVR = np.vstack((prediccion_SVR_pozo))
ax[1].imshow(Vol_arcilla_SVR, aspect = 'auto', cmap = 'jet', extent=[0,1,max(Arcilla_BLIND['Profundidad']), min(Arcilla_BLIND['Profundidad'])])
ax[1].set_title('Vsh Predicción')

fig.tight_layout()
plt.show()

#_______________________________Predecir datos faltantes_________________________________
#Escalar los datos de: registro_inc_Vsh
X_inc_Vsh = registro_inc_Vsh.drop(['Pozo','Profundidad','Permeabilidad','Volumen_arcilla'], axis = 1)

escalar = preprocessing.StandardScaler().fit(X_inc_Vsh)
inc_Vsh_escalado = escalar.transform(X_inc_Vsh)
inc_Vsh_escalado = pd.DataFrame(inc_Vsh_escalado)

prediccion_SVR_pozo = Modelo_SVR.predict(inc_Vsh_escalado)
prediccion_SVR_pozo = np.round(prediccion_SVR_pozo,4)

#Generar nuevos DataFrame
registro_inc_Vsh = registro_inc_Vsh.assign(Volumen_arcilla = prediccion_SVR_pozo)
registro_completo = registro_completo.append(registro_inc_Vsh)
registro_completo.to_csv('registro_completo.csv')
print(registro_completo.info())
#Graficar datos
plt.figure(figsize = (8, 5))
plt.scatter(registro_completo['Porosidad'], registro_completo['Saturacion_agua'], c = registro_completo['Volumen_arcilla'], s = 34, cmap = 'jet')
plt.xlabel(r'$\phi$')
plt.ylabel('Sw')
plt.title('Porosidad vs Saturación de agua [Datos completos]')
plt.colorbar(label = 'Volumen de arcilla')
plt.show()"""