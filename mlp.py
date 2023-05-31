import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models                   import Sequential
from keras.layers                   import Dense, Dropout
from keras.callbacks                import EarlyStopping
from sklearn.metrics                import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from time                           import time
from funciones_modelos              import F_and_T_Rates, predictions, plt_roc

t_i = time()

#============================================================
# Cargamos el dataset
#============================================================

print('Cargando el dataset ...\n')
X_train = pd.read_csv("./data/x_train.csv")
X_test = pd.read_csv("./data/x_test.csv")
y_test = pd.read_csv("./data/y_test.csv", index_col = False)
y_train = pd.read_csv("./data/y_train.csv")

y_train = y_train['Category']
y_test = y_test['Category']

y_test_col = y_test
y_train_col = y_train
y_train = pd.get_dummies(y_train,prefix="cat")


#============================================================
# Creamos, entrenamos y testeamos el modelo
#============================================================

print('Entrenando el modelo ...')

shape = X_train.shape[1]
mlp = Sequential()
mlp.add(Dense(128, activation='softsign', input_dim= shape))
mlp.add(Dense(128, activation='relu'))
mlp.add(Dense(64, activation='relu'))
mlp.add(Dense(64, activation='relu'))
# mlp.add(Dense(64, activation='softsign', input_dim= shape))
# mlp.add(Dense(64, activation='softsign', input_dim= shape))
mlp.add(Dense(64, activation='relu'))
mlp.add(Dense(64, activation='relu'))
mlp.add(Dense(64, activation='relu'))
mlp.add(Dense(64, activation='relu'))
mlp.add(Dense(y_train.shape[1], activation='softmax'))
mlp.summary()
mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
mlp.fit(X_train, y_train, epochs=50, verbose=1,callbacks=[monitor], batch_size=128)

print('Modelo entrenado\n')

print('Resultados del train ...')
pred_train = mlp.predict(X_train)
training = []
for i in range(len(pred_train)):
    training.append(np.argmax(pred_train[i]))
print('Accuracy for MLP - Train:\t{}\n'.format(accuracy_score(training, y_train_col)))

print('Resultados del test ...')
pred_test = mlp.predict(X_test)
testing= []
for i in range(len(pred_test)):
    testing.append(np.argmax(pred_test[i]))

print('Accuracy for MLP - Test:\t{}\n'.format(accuracy_score(testing, y_test_col)))

print("Creando la matriz de confusion ...")
fig = plt.figure(figsize=(11,11))
c_matrix = confusion_matrix(y_test_col, testing)
sns.heatmap(c_matrix, cmap="YlOrRd", annot=True)
plt.title("Matriz de confusión del MLP")
fig.savefig("./img/MLP/CM_MLP.png", dpi=300)
print("Matriz de confusion para MLP guardada.\n")

#============================================================
# TPR, TNR, FPR Y FNR
#============================================================

print('Calculando TPR, TNR, FPR y FNR ...')
F_and_T_Rates(c_matrix)
print('')

#============================================================
# Metricas
#============================================================

print('Clasification Report para MLP:')
print(classification_report(y_test_col, testing))

print("Precision MLP:\t{}".format(precision_score(y_test_col, testing, average='weighted')))
print("Recall MLP:\t{}".format(recall_score(y_test_col, testing, average='weighted')))
print("F1 Score MLP:\t{}\n".format(f1_score(y_test_col, testing, average='weighted')))

#============================================================
# Dataset con todas las predicciones y estadisticas
#============================================================

print("Creando el dataset de probabilidades y prediccion ...")
df_pred = predictions(pred_test, y_test_col)
df_pred.to_csv('./analysis/MLP.csv')
print('Dataset creado\n')

#============================================================
# Curvas ROC para el MLP
#============================================================

print('Dibujando las curvas ROC ...')
plt = plt_roc(pred_test, df_pred)
plt.savefig("./img/MLP/ROC_mlp.png", dpi=300)
print('Curvas ROC pintadas y guardadas\n')

#============================================================
# Guardamos el modelo
#============================================================

print('Guardando el modelo ...')
mlp.save('./saved_model/mlp_model.h5')
print('Modelo guardado correctamente.\n')

t_f= time()
print('Tiempo de ejecucion: ', t_f-t_i, ' segundos = ', (t_f-t_i)/60, 'minutos')