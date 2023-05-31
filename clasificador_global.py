import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.discriminant_analysis  import StandardScaler
from sklearn.metrics                import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from time                           import time
from funciones_modelos              import F_and_T_Rates, predictions, plt_roc

t_i = time()

#============================================================
# Cargamos el dataset
#============================================================

print('Cargando el dataset ...\n')

X_test = pd.read_csv("./data/x_test.csv")
y_test = pd.read_csv("./data/y_test.csv", index_col = False)
y_test = y_test['Category']

#============================================================
# Modelos Individuales y sus predicciones
#============================================================

#-----------------------
# Random Forest
#-----------------------
rf = joblib.load('./saved_model/rf_model.pkl')
rf_pred = rf.predict_proba(X_test)
rf_res_tot = rf.predict(X_test)

#-----------------------
# SVM
#-----------------------
svm = joblib.load('./saved_model/svm_model.pkl')
scaler = StandardScaler().fit(X_test)
X_test_svm = scaler.transform(X_test)
svm_pred = svm.predict_proba(X_test_svm)
svm_res_tot = svm.predict(X_test_svm)

#-----------------------
# KNN
#-----------------------
knn = joblib.load('./saved_model/knn_model.pkl')
knn_pred = knn.predict_proba(X_test)
knn_res_tot = knn.predict(X_test)

#-----------------------
# LSTM
#-----------------------
lstm = tf.keras.models.load_model('./saved_model/lstm_model.h5')
X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))
lstm_pred = lstm.predict(X_test_lstm)
lstm_res_tot= []
for i in range(len(lstm_pred)):
    lstm_res_tot.append(np.argmax(lstm_pred[i]))

#============================================================
# Clasificador global
#============================================================

# Matriz con los TPR de los modelos a usar
tprs = {
        'KNN':[0.767, 0.988, 1.0, 0.988, 0.875, 0.930, 0.997, 0.959, 1.0, 0.959, 0.965, 0.953],
        'SVM': [0.923, 0.979, 1.0, 0.991, 0.822, 0.940, 1.0, 0.963, 0.999, 0.828, 0.973,0.952],
        'RF': [0.934, 0.964, 0.998, 0.974, 0.887, 0.935, 0.999, 0.934, 0.999, 0.822, 0.919, 0.915],
        'LSTM': [0.925, 0.967, 1.0, 0.994, 0.818, 0.931, 0.998, 0.957, 0.999, 0.848, 0.945, 0.953],
}

# Creamos un dataset donde guardar las ponderaciones que les daremos a los TPR
df_ponderaciones = pd.DataFrame(columns=['RF_pond', 'SVM_pond', 'KNN_pond', 'LSTM_pond'])

# Calculamos las ponderaciones y rellenamos el dataset
for i in range(len(tprs['RF'])):
    pot = 10
    tot = (pow(tprs['RF'][i], pot) + pow(tprs['SVM'][i], pot) + pow(tprs['KNN'][i], pot)+ pow(tprs['LSTM'][i], pot))
    rf_pond = pow(tprs['RF'][i], pot) / tot
    svm_pond = pow(tprs['SVM'][i], pot) / tot
    knn_pond = pow(tprs['KNN'][i], pot) / tot
    lstm_pond = pow(tprs['LSTM'][i], pot) / tot
    df_ponderaciones.loc[i] = [rf_pond, svm_pond, knn_pond, lstm_pond]

print(df_ponderaciones)

# Creamos las matrices que guardan las predicciones ponderadas
rf_pred_pond = np.zeros_like(rf_pred)
svm_pred_pond = np.zeros_like(svm_pred)
knn_pred_pond = np.zeros_like(knn_pred)
lstm_pred_pond = np.zeros_like(lstm_pred)

# Multiplicar cada fila de la predicción por el valor de su ponderación
for i in range(len(rf_pred)):
    index_rf = rf_res_tot[i]
    index_svm = svm_res_tot[i]
    index_knn = knn_res_tot[i]
    index_lstm = lstm_res_tot[i]
    rf_pred_pond[i] = rf_pred[i] * df_ponderaciones['RF_pond'][index_rf]
    svm_pred_pond[i] = svm_pred[i] * df_ponderaciones['SVM_pond'][index_svm]
    knn_pred_pond[i] = knn_pred[i] * df_ponderaciones['KNN_pond'][index_knn]
    lstm_pred_pond[i] = lstm_pred[i] * df_ponderaciones['LSTM_pond'][index_lstm]

pred_total = rf_pred_pond + svm_pred_pond + knn_pred_pond + lstm_pred_pond

print(pred_total)

#============================================================
# Dataset con todas las predicciones y estadisticas
#============================================================

print(pred_total.dtype)
print(y_test.dtype)

df_pred = predictions(pred_total, y_test)

print('Accuracy for Global Classifier - Test:\t{}\n'.format(accuracy_score(df_pred['Prediction'], y_test)))

#============================================================
# Matriz de confusión
#============================================================

print("Creando la matriz de confusion ...")
fig = plt.figure(figsize=(11,11))
c_matrix = confusion_matrix(y_test, df_pred['Prediction'])
sns.heatmap(c_matrix, cmap="YlOrRd", annot=True)
plt.title("Matriz de Confusión del Clasificador Global")
fig.savefig("./img/Clasificador_Global/CM_CG.png", dpi=300)
print("Matriz de confusion para el Clasificador Global guardada.\n")

#============================================================
# TPR, TNR, FPR Y FNR
#============================================================

print('Calculando TPR, TNR, FPR y FNR ...')
F_and_T_Rates(c_matrix)
print('')

#============================================================
# Precision, Recall y F1-Score
#============================================================

print('Clasification Report para el Clasificador Global:')
print(classification_report(y_test, df_pred['Prediction']))

print("Precision el Clasificador Global:\t{}".format(precision_score(y_test, df_pred['Prediction'], average='weighted')))
print("Recall el Clasificador Global:\t{}".format(recall_score(y_test, df_pred['Prediction'], average='weighted')))
print("F1 Score el Clasificador Global:\t{}\n".format(f1_score(y_test, df_pred['Prediction'], average='weighted')))

#============================================================
# Dataset con todas las predicciones y estadisticas
#============================================================

print("Creando el dataset de probabilidades y prediccion ...")
pred_prob = rf.predict_proba(X_test)
df_pred = predictions(pred_prob, y_test)
df_pred.to_csv('./analysis/CG.csv')
print('Dataset creado\n')

#============================================================
# Curvas ROC para el Clasificador Global
#============================================================

print('Dibujando las curvas ROC ...')
plt = plt_roc(pred_prob, df_pred)
plt.savefig("./img/Clasificador_Global/ROC_CG.png", dpi=300)
print('Curvas ROC pintadas y guardadas\n')