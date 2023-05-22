import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models                   import Sequential
from keras.layers                   import Dense, LSTM, Dropout
from keras.wrappers.scikit_learn    import KerasClassifier
from keras.callbacks                import EarlyStopping
from sklearn.metrics                import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score
from sklearn.model_selection        import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing          import label_binarize
from sklearn.feature_selection      import SelectKBest, mutual_info_classif
from scipy.stats                    import zscore
from time                           import time
from itertools                      import cycle
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

X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

y_test_col = y_test
y_train_col = y_train
y_train = pd.get_dummies(y_train,prefix="cat")

#============================================================
# Creamos, entrenamos y testeamos el modelo
#============================================================

print('Entrenando el modelo ...')

lstm = Sequential()
lstm.add(LSTM(128, activation='relu', input_shape=(None, X_train.shape[2]), return_sequences=True))
lstm.add(LSTM(128, activation='relu', return_sequences=True))
lstm.add(LSTM(128, activation='relu', return_sequences=True))
lstm.add(LSTM(128, activation='relu'))
lstm.add(Dense(y_train.shape[1], activation='softmax'))
lstm.summary()
lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
monitor = EarlyStopping(monitor='loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
lstm.fit(X_train, y_train, epochs=100, verbose=1,callbacks=[monitor], batch_size = 64)

print('Modelo entrenado\n')

print('Resultados del train ...')
pred_train = lstm.predict(X_train)
training = []
for i in range(len(pred_train)):
    training.append(np.argmax(pred_train[i]))
print('Accuracy for LSTM - Train:\t{}\n'.format(accuracy_score(training, y_train_col)))

print('Resultados del test ...')
pred_test = lstm.predict(X_test)
testing= []
for i in range(len(pred_test)):
    testing.append(np.argmax(pred_test[i]))

print('Accuracy for LSTM - Test:\t{}\n'.format(accuracy_score(testing, y_test_col)))

print("Creando la matriz de confusion ...")
fig = plt.figure(figsize=(11,11))
c_matrix = confusion_matrix(y_test_col, testing)
sns.heatmap(c_matrix, cmap="RdPu", annot=True)
plt.title("Matriz de Confusión del LSTM")
fig.savefig("./img/LSTM/CM_LSTM.png", dpi=300)
print("Matriz de confusion para LSTM guardada.\n")

#============================================================
# TPR, TNR, FPR Y FNR
#============================================================

print('Calculando TPR, TNR, FPR y FNR ...')
F_and_T_Rates(c_matrix)
print('')

#============================================================
# Metricas
#============================================================

print('Clasification Report para LSTM:')
print(classification_report(y_test_col, testing))

print("Precision LSTM:\t{}".format(precision_score(y_test_col, testing, average='weighted')))
print("Recall LSTM:\t{}".format(recall_score(y_test_col, testing, average='weighted')))
print("F1 Score LSTM:\t{}\n".format(f1_score(y_test_col, testing, average='weighted')))

#============================================================
# Dataset con todas las predicciones y estadisticas
#============================================================

print("Creando el dataset de probabilidades y prediccion ...")
df_pred = predictions(pred_test, y_test_col)
df_pred.to_csv('./analysis/LSTM.csv')
print('Dataset creado\n')

#============================================================
# Curvas ROC para el LSTM
#============================================================

print('Dibujando las curvas ROC ...')
plt = plt_roc(pred_test, df_pred)
plt.savefig("./img/LSTM/ROC_lstm.png", dpi=300)
print('Curvas ROC pintadas y guardadas\n')

#============================================================
# Guardamos el modelo
#============================================================

print('Guardando el modelo ...')
lstm.save('./saved_model/lstm_model.h5')
print('Modelo guardado correctamente.\n')

t_f= time()
print('Tiempo de ejecucion: ', t_f-t_i, ' segundos = ', (t_f-t_i)/60, 'minutos')