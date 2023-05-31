import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics                import roc_curve, auc
from sklearn.model_selection        import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing          import label_binarize
from itertools                      import cycle


#===============================================================================================================
# GENERALES ALGORITMOS
#===============================================================================================================

#-------------------------------------------------------
# Calcular los True y False Rates
#-------------------------------------------------------

def F_and_T_Rates(c_matrix):
    FP = c_matrix.sum(axis=0) - np.diag(c_matrix)   # Suma las columnas
    FN = c_matrix.sum(axis=1) - np.diag(c_matrix)   # Suma las filas
    TP = np.diag(c_matrix)
    TN = c_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Recall or True Positive Rate
    TPR = TP/(TP+FN)
    print('')
    print('TPR :')
    for i in range (12):
        print('Class ', i, ' :', TPR[i])
    # True negative rate
    TNR = TN/(TN+FP)
    print('') 
    print('TNR :')
    for i in range (12):
        print('Class ', i, ' :', TNR[i])
    # False positive rate
    FPR = FP/(FP+TN)
    print('')
    print('FPR :')
    for i in range (12):
        print('Class ', i, ' :', FPR[i])
    # False negative rate
    FNR = FN/(TP+FN)
    print('')
    print('FNR :')
    for i in range (12):
        print('Class ', i, ' :', FNR[i])

#-------------------------------------------------------
# Generar un dataset con las predicciones y probabilidades
#-------------------------------------------------------

def predictions(pred_prob, y_test):
    
    df_pred = pd.DataFrame(columns=('Prob0','Prob1','Prob2','Prob3','Prob4','Prob5','Prob6','Prob7','Prob8','Prob9','Prob10','Prob11',
                                    'Prediction', 'Probability','Real Class', 'Decision'))
    df_pred['Real Class'] = y_test

    for i in range(12):
        prob_i = []
        for x in range(len(pred_prob)):
            prob_i.append(pred_prob[x][i])
        df_pred['Prob'+ str(i)] = prob_i

    type_argmax= []
    prob_argmax = []
    for i in range(len(pred_prob)):
        type_argmax.append(np.argmax(pred_prob[i]))
        prob_argmax.append(pred_prob[i][np.argmax(pred_prob[i])])
    df_pred['Prediction'] = type_argmax
    df_pred['Probability'] = prob_argmax

    decision = []
    for index, row in df_pred.iterrows():
        if row['Prediction'] == row['Real Class']:
            decision.append(1)
        else:
            decision.append(0)
    df_pred['Decision'] = decision

    return df_pred

#-------------------------------------------------------
# Funcion para calcular y dibujar las curvas ROC
#-------------------------------------------------------

def plt_roc(y_score, df_pred):
    
    y_ts = label_binarize(df_pred['Real Class'].values, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    n_classes = y_ts.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_ts[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_ts.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
            label='curva ROC media-micro (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='magenta', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='curva ROC media-macro (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='lime', linestyle=':', linewidth=4)

    colors = cycle(['paleturquoise', 'salmon', 'peru', 'lightpink', 'aqua', 'darkorange', 'cornflowerblue', 'violet', 'slategray', 'yellowgreen', 'blue', 'khaki'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=lw,
                label='Curva ROC clase {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', linewidth=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Característica operativa del receptor (ROC) para multiclase')
    plt.legend(loc="lower right")
    #plt.show()

    return plt

#===============================================================================================================
# ALGORITMOS (SVM, RF, KNN)
#===============================================================================================================

#-------------------------------------------------------
# Hyper-parameters Cross-validation con GridSearchCV
#-------------------------------------------------------

def CV_GridSearch(X_train, y_train, X_test, y_test, model, param_grid):

    print(param_grid)
    
    GridSearch = GridSearchCV(estimator = model, param_grid = param_grid, verbose=2)

    GridSearch.fit(X_train, y_train)

    df_cv_results = pd.DataFrame(GridSearch.cv_results_)
    df_cv_results.sort_values('rank_test_score', inplace = True)

    # Resultados del GridSearch
    print("Best score: ", GridSearch.best_estimator_)
    print("Best params: ", GridSearch.best_params_)

    print (f'Train Accuracy - : {GridSearch.score(X_train,y_train):.3f}')
    print (f'Test Accuracy - : {GridSearch.score(X_test,y_test):.3f}')

    return df_cv_results

#-------------------------------------------------------
# Hyper-parameters Cross-validation con RandomizedSearchCV
#-------------------------------------------------------

def CV_RandomizedSearch(X_train, y_train, X_test, y_test, model, param_grid):

    print(param_grid)
    
    RandomGrid = RandomizedSearchCV(estimator = model, param_distributions = param_grid, verbose=2)

    RandomGrid.fit(X_train, y_train)

    df_cv_results = pd.DataFrame(RandomGrid.cv_results_)
    df_cv_results.sort_values('rank_test_score', inplace = True)

    print("Best score: ", RandomGrid.best_estimator_)
    print("Best params: ", RandomGrid.best_params_)

    print (f'Train Accuracy - : {RandomGrid.score(X_train,y_train):.3f}')
    print (f'Test Accuracy - : {RandomGrid.score(X_test,y_test):.3f}')

    return df_cv_results

#-------------------------------------------------------
# Hyper-parameters Bayesian techniques
#-------------------------------------------------------
