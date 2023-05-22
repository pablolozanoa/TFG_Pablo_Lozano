import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.discriminant_analysis  import StandardScaler
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

X_test = pd.read_csv("./data/x_test.csv")
y_test = pd.read_csv("./data/y_test.csv", index_col = False)
y_test = y_test['Category']

#============================================================
# Clasificador global
#============================================================

#-----------------------
# Random Forest
#-----------------------

rf = joblib.load('./saved_model/rf_model.pkl')
rf_pred = rf.predict_proba(X_test)
print(rf_pred)
print(np.shape(rf_pred))

#-----------------------
# SVM
#-----------------------

svm = joblib.load('./saved_model/svm_model.pkl')
scaler = StandardScaler().fit(X_test)
X_test_svm = scaler.transform(X_test)
svm_pred = svm.predict_proba(X_test_svm)
print(svm_pred)
print(np.shape(svm_pred))

#-----------------------
# KNN
#-----------------------

knn = joblib.load('./saved_model/knn_model.pkl')
knn_pred = knn.predict_proba(X_test)
print(knn_pred)
print(np.shape(knn_pred))

#-----------------------
# MLP
#-----------------------

# mlp = tf.keras.models.load_model('./saved_model/mlp_model.h5')
# mlp_pred = mlp.predict(X_test)
# print(mlp_pred)
# print(np.shape(mlp_pred))

#-----------------------
# LSTM
#-----------------------

# lstm = tf.keras.models.load_model('./saved_model/lstm_model.h5')
# lstm_pred = lstm.predict(X_test)
# print(lstm_pred)
# print(np.shape(lstm_pred))



pred_total = rf_pred + svm_pred + knn_pred 
print(pred_total)

#============================================================
# Dataset con todas las predicciones y estadisticas
#============================================================

print(pred_total.dtype)
print(y_test.dtype)

df_pred = predictions(pred_total, y_test)
print('Dataset creado\n')

print('Accuracy for Global Classifier - Test:\t{}\n'.format(accuracy_score(df_pred['Prediction'], y_test)))
