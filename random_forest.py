import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.discriminant_analysis  import StandardScaler
from sklearn.ensemble               import RandomForestClassifier
from sklearn.metrics                import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score, recall_score, precision_score
from sklearn.model_selection        import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing          import label_binarize
from sklearn.feature_selection      import SelectKBest, mutual_info_classif, chi2
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

#============================================================
# Optimizamos los hiperparámetros
#============================================================

print('Optimizando los hiperparametros ...')

model = RandomForestClassifier()
print(model.get_params(),'\n')

# Vamos variando el param_grid segun que parametros queramos ajustar y según si es Grid o Randomized

param_grid = {'criterion': ('gini', 'entropy'),
                    'max_depth': [50],
                    'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 15)],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf' : [1, 2],
                    'bootstrap' : [True, False],
                    'max_features' : ['sqrt', 'log2', None],
                    }

#------------------------
# Grid Search
#------------------------
# df_cv_results = CV_GridSearch(X_train, y_train, X_test, y_test, model, param_grid)
# df_cv_results.to_csv("./cv_results/RF/RF_GridSearchCV_1")

#------------------------
# Random Search
#------------------------
# df_cv_results = CV_RandomizedSearch(X_train, y_train, X_test, y_test, model,  param_grid)
# df_cv_results.to_csv("./cv_results/RF/RF_RandomizedSearchCV_1")

#------------------------
# Bayesian
#------------------------
def gbm_cl_bo(max_samples,n_estimators,max_features):
    params_gbm = {
        'max_samples': max_samples,
        'max_features':max_features,
        'n_estimators':int(n_estimators)
    }
    scores = cross_val_score(RandomForestClassifier(random_state=123, **params_gbm),
                                X_train, y_train, scoring= 'accuracy', cv=5).mean()
    score = scores.mean()
    return score

params_gbm ={
            'max_samples':(0.5,1),
            'max_features':(0.5,1),
            'n_estimators':(100,200)
            }
# gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm, random_state=111)
# gbm_bo.maximize(init_points=20, n_iter=40)

# params_gbm = gbm_bo.max['params']
# print(params_gbm)

#============================================================
# Entrenamos y testeamos el modelo
#============================================================

print('Entrenando el modelo ...')
# Train the model for best result in cross validation
rf = RandomForestClassifier(#criterion='entropy',
                            #bootstrap= False,
                            #max_features= 'sqrt',
                            #max_depth= 50,
                            n_estimators= 4,
                            #verbose= 2,
                            random_state= 1
                            )
print(rf.get_params())

rf.fit(X_train, y_train)
print('Modelo entrenado\n')

print('Resultados del train ...')
pred_train = rf.predict(X_train)
print('Accuracy for Random Forest - Train:\t{}\n'.format(accuracy_score(pred_train, y_train)))

print('Resultados del test ...')
pred = rf.predict(X_test)
print('Accuracy for Random Forest Classifier - Test:\t{}\n'.format(accuracy_score(pred, y_test)))

print("Creando la matriz de confusion ...")
fig = plt.figure(figsize=(11,11))
c_matrix = confusion_matrix(y_test, pred)
sns.heatmap(c_matrix, cmap="PuBu", annot=True)
plt.title("Matriz de Confusión del Random Forest")
fig.savefig("./img/RF/CM_RF.png", dpi=300)
print("Matriz de confusion para Random Forest guardada.\n")

#============================================================
# TPR, TNR, FPR Y FNR
#============================================================

print('Calculando TPR, TNR, FPR y FNR ...')
F_and_T_Rates(c_matrix)
print('')

#============================================================
# Metricas
#============================================================

print('Clasification Report para Random Forest:')
print(classification_report(y_test, pred))

print("Precision Random Forest:\t{}".format(precision_score(y_test, pred, average='weighted')))
print("Recall Random Forest:\t{}".format(recall_score(y_test, pred, average='weighted')))
print("F1 Score Random Forest:\t{}\n".format(f1_score(y_test, pred, average='weighted')))

#============================================================
# Dataset con todas las predicciones y estadisticas
#============================================================

print("Creando el dataset de probabilidades y prediccion ...")
pred_prob = rf.predict_proba(X_test)
df_pred = predictions(pred_prob, y_test)
df_pred.to_csv('./analysis/RF.csv')
print('Dataset creado\n')

#============================================================
# Curvas ROC para el Random Forest
#============================================================

print('Dibujando las curvas ROC ...')
plt = plt_roc(pred_prob, df_pred)
plt.savefig("./img/RF/ROC_rf.png", dpi=300)
print('Curvas ROC pintadas y guardadas\n')

#============================================================
# Guardamos el modelo
#============================================================

print('Guardando el modelo ...')
joblib.dump(rf, './saved_model/rf_model.pkl')
print('Modelo guardado correctamente.\n')

t_f= time()
print('Tiempo de ejecucion: ', t_f-t_i, ' segundos = ', (t_f-t_i)/60, 'minutos')