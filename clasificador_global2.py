import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from scipy.stats                    import mode
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

df_ensemble = pd.DataFrame(columns=['rf', 'svm', 'knn', 'prediction', 'real'])
df_ensemble['real'] = y_test

#-----------------------
# Random Forest
#-----------------------

rf = joblib.load('./saved_model/rf_model.pkl')
rf_pred = rf.predict(X_test)
df_ensemble['rf'] = rf_pred

#-----------------------
# SVM
#-----------------------

svm = joblib.load('./saved_model/svm_model.pkl')
scaler = StandardScaler().fit(X_test)
X_test_svm = scaler.transform(X_test)
svm_pred = svm.predict(X_test_svm)
df_ensemble['svm'] = svm_pred

#-----------------------
# KNN
#-----------------------

knn = joblib.load('./saved_model/knn_model.pkl')
knn_pred = knn.predict(X_test)
df_ensemble['knn'] = knn_pred

#-----------------------
# MLP
#-----------------------



#-----------------------
# LSTM
#-----------------------


#============================================================
# Dataset con todas las predicciones y estadisticas
#============================================================

print(df_ensemble)

for index, row in df_ensemble.iterrows():
    rf_value = row['rf']
    svm_value = row['svm']
    knn_value = row['knn']
    
    most_common = np.argmax(np.bincount([rf_value, svm_value, knn_value]))
    df_ensemble.at[index, 'prediction'] = most_common
    
    # if svm_value == 6:
    #     df_ensemble.at[index, 'prediction'] = 6
    # elif svm_value == 8:
    #     df_ensemble.at[index, 'prediction'] = 8
    # elif svm_value == 2:
    #     df_ensemble.at[index, 'prediction'] = 2
    # elif knn_value == 3:
    #     df_ensemble.at[index, 'prediction'] = 3
    # elif knn_value == 1:
    #     df_ensemble.at[index, 'prediction'] = 1
    # elif knn_value == 11:
    #     df_ensemble.at[index, 'prediction'] = 11
    # elif knn_value == 9:
    #     df_ensemble.at[index, 'prediction'] = 9
    # elif svm_value == 10:
    #     df_ensemble.at[index, 'prediction'] = 10
    # elif rf_value == 0:
    #     df_ensemble.at[index, 'prediction'] = 0
    # else:
    #     most_common = np.argmax(np.bincount([rf_value, svm_value, knn_value]))
    #     df_ensemble.at[index, 'prediction'] = most_common

df_ensemble['prediction'] = df_ensemble['prediction'].astype(int)

print(df_ensemble)

print(df_ensemble['prediction'].dtype)
print(np.unique(df_ensemble['prediction']))
print(df_ensemble['real'].dtype)
print(np.unique(df_ensemble['real']))

print('Accuracy for Global Classifier - Test:\t{}\n'.format(accuracy_score(df_ensemble['prediction'], df_ensemble['real'])))
