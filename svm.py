import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm                    import SVC
from sklearn.metrics                import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score
from sklearn.model_selection        import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing          import label_binarize, StandardScaler
from sklearn.feature_selection      import SelectKBest, mutual_info_classif
from scipy.stats                    import zscore
from bayes_opt                      import BayesianOptimization
from time                           import time
from itertools                      import cycle
from funciones_modelos              import F_and_T_Rates, predictions, plt_roc, CV_GridSearch, CV_RandomizedSearch

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

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#============================================================
# Optimizamos los hiperparámetros
#============================================================

print('Optimizando los hiperparametros ...')

model = SVC()
print(model.get_params(),'\n')

# Vamos variando el param_grid segun que parametros queramos ajustar y según si es Grid o Randomized

param_grid = {'C':[int(x) for x in np.linspace(start = 1, stop = 200, num = 20)],
                    'kernel':('rbf', 'linear', 'poly', 'sigmoid'), 
                    'max_iter': [int(x) for x in np.linspace(start = 1, stop = 100000, num = 20)],
                    'gamma': ('scale', 'auto'),
                    'decision_function_shape':('ovo', 'ovr'),
                    'shrinking': [True, False],
                    'probability' : [True, False]
                    } 

#------------------------
# Grid Search
#------------------------
# df_cv_results = CV_GridSearch(X_train, y_train, X_test, y_test, model, param_grid)
# df_cv_results.to_csv("./cv_results/SVM/SVM_GridSearchCV_1")

#------------------------
# Random Search
#------------------------
# df_cv_results = CV_RandomizedSearch(X_train, y_train, X_test, y_test, model,  param_grid)
# df_cv_results.to_csv("./cv_results/SVM/SVM_RandomizedSearchCV_1")

#------------------------
# Bayesian
#------------------------
def gbm_cl_bo(C,max_iter):
    params_gbm = {
        'C': C,
        'max_iter':int(max_iter),
    }
    scores = cross_val_score(SVC(random_state=123, **params_gbm),
                                X_train, y_train, scoring= 'accuracy', cv=5).mean()
    score = scores.mean()
    return score

params_gbm ={
            'C':(1, 1000),
            'max_iter':(1, 800000),
            }

# gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm, random_state=111)
# gbm_bo.maximize(init_points=20, n_iter=40)

# params_gbm = gbm_bo.max['params']
# print(params_gbm)

#============================================================
# Entrenamos y testeamos el modelo
#============================================================

print('Entrenando el modelo ...')
svm = SVC(kernel='rbf', 
            gamma='scale', 
            C=100,
            max_iter=150000,
            #verbose= 2, 
            probability=True,
            random_state=1
            )
print(svm.get_params())

svm.fit(X_train, y_train)
print('Modelo entrenado\n')

print('Resultados del train ...')
pred_train = svm.predict(X_train)
print('Accuracy for SVM - Train:\t{}\n'.format(accuracy_score(pred_train, y_train)))

print('Resultados del test ...')
pred = svm.predict(X_test)
print('Accuracy for SVM Classifier - Test:\t{}\n'.format(accuracy_score(pred, y_test)))

print("Creando la matriz de confusion ...")
fig = plt.figure(figsize=(11,11))
c_matrix = confusion_matrix(y_test, pred)
sns.heatmap(c_matrix, cmap="YlGnBu", annot=True)
plt.title("Confusion Matrix SVM Classifier")
fig.savefig("./img/SVM/CM_SVM.png", dpi=300)
print("Matriz de confusion para SVM guardada.\n")
#print(c_matrix)

#============================================================
# TPR, TNR, FPR Y FNR
#============================================================

print('Calculando TPR, TNR, FPR y FNR ...')
F_and_T_Rates(c_matrix)
print('')

#============================================================
# Metricas
#============================================================

print('Clasification Report para SVM:')
print(classification_report(y_test, pred))

print("Precision SVM:\t{}".format(precision_score(y_test, pred, average='weighted')))
print("Recall SVM:\t{}".format(recall_score(y_test, pred, average='weighted')))
print("F1 Score SVM:\t{}\n".format(f1_score(y_test, pred, average='weighted')))

#============================================================
# Dataset con todas las predicciones y estadisticas
#============================================================

print("Creando el dataset de probabilidades y prediccion ...")
pred_prob = svm.predict_proba(X_test)
df_pred = predictions(pred_prob, y_test)
df_pred.to_csv('./analysis/SVM.csv')
print('Dataset creado\n')

#============================================================
# Curvas ROC para el SVM
#============================================================

print('Dibujando las curvas ROC ...')
plt = plt_roc(pred_prob, df_pred)
plt.savefig("./img/SVM/ROC_svm.png", dpi=300)
print('Curvas ROC pintadas y guardadas\n')

#============================================================
# Guardamos el modelo
#============================================================

print('Guardando el modelo ...')
joblib.dump(svm, './saved_model/svm_model.pkl')
print('Modelo guardado correctamente.\n')

t_f= time()
print('Tiempo de ejecucion: ', t_f-t_i, ' segundos = ', (t_f-t_i)/60, 'minutos')