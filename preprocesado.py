import os
import pandas                   as pd
import numpy                    as np
import matplotlib.pyplot        as plt
import seaborn                  as sns
from os.path                    import isfile, join
from sklearn.preprocessing      import LabelEncoder
from collections                import Counter
from scipy.stats                import zscore
from sklearn.model_selection    import train_test_split
from sklearn.feature_selection  import mutual_info_classif
from sklearn.preprocessing      import MinMaxScaler
from imblearn.over_sampling     import SMOTE
from imblearn.under_sampling    import NearMiss, TomekLinks, EditedNearestNeighbours


#============================================================
# Unión de los datasets
#============================================================

print("Juntando todos los Datasets ...")

def ficheros(ruta):
    contenido = os.listdir(ruta)
    archivos = [nombre for nombre in contenido if isfile(join(ruta, nombre))]
    return archivos

files = ficheros('./original_dataset/before_reboot')
files.remove('No_Category_before_reboot_Cat.csv')
files.remove('Zero_Day_before_reboot_Cat.csv')
files_2 = ficheros('./original_dataset/after_reboot')
files_2.remove('No_Category_after_reboot_Cat.csv')
files_2.remove('Zero_Day_after_reboot_Cat.csv')
print(files)

df = pd.read_csv("./original_dataset/before_reboot/"+ files[0], na_values=['NA', '?', 'NaN'])
df_2 = pd.read_csv("./original_dataset/after_reboot/"+ files_2[0], na_values=['NA', '?', 'NaN'])
df = pd.concat([df, df_2], ignore_index= True)

for x in range(1, len(files)):
    df_next = pd.read_csv("./original_dataset/before_reboot/"+ files[x], na_values=['NA', '?', 'NaN'])
    df_next_2 = pd.read_csv("./original_dataset/after_reboot/"+ files_2[x], na_values=['NA', '?', 'NaN'])
    df = pd.concat([df, df_next, df_next_2], ignore_index= True)

print(np.shape(df))

print("Los datasets han sido unidos en uno correctamente", "\n")

#============================================================
# Comprobación Missing Values ('NaN', 'NA', '?)
#============================================================

print("Comprobando Missing Values ...")

# Comprobamos si existe alguna fila que tenga alguna feature NaN
filas_faltantes = df[df.isna().any(axis=1)]

# Si no hay ninguna Nan no hacemos nada, si las hay las borramos
if len(filas_faltantes) == 0:
    print("No hay ningun Missing-Value", "\n")
else:
    indices = filas_faltantes.index
    df = df.drop(indices)
    print("Se han eliminado las filas que contenían algún Missing-Value", "\n")

#============================================================
# Dropping constant Features
#============================================================
# Esto es parte del Feature selection, pero de cara al uso de proximas funciones nos combiene eliminar ya categorías constantes

print("Eliminando caracteristicas constantes ...")

# Seleccionar las características numéricas
numeric_features = df.select_dtypes(include=[np.number])

# Identificar las características constantes
constant_features = []
for column in numeric_features.columns:
    if numeric_features[column].std() == 0: # Si la desviación típica == 0
        constant_features.append(column)

# Imprimir las características constantes
for constant_feature in constant_features:
    df.drop(constant_feature, axis= 1, inplace=True)
    print("La caracteristica {} ha sido eliminada por ser constante".format(constant_feature))

print('Características constantes eliminadas\n')

#============================================================
# Análisis de los valores No numéricos
#============================================================

print("Analizando los valores No Numericos ...")

# Separamos las features numéricas y las no numéricas
non_num = []
num = []
for c in df.columns:
    if df[c].dtypes not in ('int64', 'float64'):
        non_num.append(c)
    else: num.append(c)
print("Non-numerical Values: ", non_num)

for item in non_num:
    print("Number of distinct values for ", item, ": ",df[item].unique().size)

# Codificamos 'Hash' y 'Categories' e imprimimos por pantalla la transformación de 'Categories'
le_hash = LabelEncoder()
df['Hash'] = le_hash.fit_transform(df['Hash'])

le_category = LabelEncoder()
df['Category'] = le_category.fit_transform(df['Category'])

for i in range(len(le_category.classes_)):
    print(i, ': ', le_category.classes_[i])
print("")

# Eliminamos la catedoria 'Family' porque no la usaremos para clasificar
df.drop('Family', axis= 1, inplace=True)

#============================================================
# Comprobación Inf Values
#============================================================

print("Comprobando Infinite values ...")

# Comprobamos si existe alguna fila que tenga alguna feature NaN
inf_values = df[np.isinf(df).any(axis=1)]

# Si no hay ninguna Nan no hacemos nada, si las hay las borramos
if len(inf_values) == 0:
    print("No hay ningun Infinite-Value", "\n")
else:
    indices = inf_values.index
    df = df.drop(indices)
    print("Se han eliminado las filas que contenían algún Infinite-Value", "\n")

#============================================================
# Análisis de los valores numéricos
#============================================================

print("Analizando los valores Numericos ...")

scaler = MinMaxScaler()

for feature in num:
    #df[feature] = zscore(df[feature])
    df[feature] = scaler.fit_transform(df[[feature]])
#df['Hash'] = zscore(df['Hash'])
df["Hash"] = scaler.fit_transform(df[["Hash"]])

print("Valores transformados usando Zscore")

# # Mezclamos las filas con la misma semilla para obtener siempre la misma mezcla
# np.random.seed(33)
# df = df.reindex(np.random.permutation(df.index))

X = df.drop('Category',axis= 1)
y = df['Category']

counter = Counter(y)
print(counter)
print([(i, counter[i] / sum(counter.values())  * 100.0) for i in counter], "\n")

#============================================================
# Estudio y eliminación (drop) de las caracteristicas (features) (Feature Selection -> FS)
#============================================================
print("Estudio de los features y seleccion...", "\n")
#---------------------------------
# Information Gain
#---------------------------------
print("Calculando ganancia de informacion ...")

ig = mutual_info_classif(X, y)

# # Imprimimos por pantalla la Ganancia de Información por cada feature
# for i, feature in enumerate(X.columns):
#     print("Ganancia de informacion para", feature, "=", ig[i])

# Dibujo de la gráfica con todas las features juntas
feat_ig = pd.Series(ig, df.columns[0:len(df.columns)-1])
fig, ax = plt.subplots(figsize=(28,11))
feat_ig.plot.bar(color ='teal')
ax.tick_params(axis='both', which='major', labelsize=5)
#plt.show()
fig.savefig("./img/information_gain/Global.png", dpi=300, bbox_inches='tight')

# Array con las features que hay que elminar porque tienen una IG muy baja
drop_features = []
for x in range(0, len(ig)):
    if ig[x] < 0.05:
        drop_features.append(X.columns[x])
        print("La caracteristica {} va a ser eliminada por tener una IG muy baja".format(X.columns[x]))

# Dibujo de las gráficas de barras según sus categorías y orden descendente
categories = ["Memory", "API","Network_Total", "Battery", "Log", "Process_total", "Hash"]
size = [20, 25, 11, 11, 11, 11, 11]
for category in categories:
    X_cadena = X.copy()
    for column in X.columns:
        if category not in column:
            X_cadena.drop(column, axis= 1, inplace=True)
    ig = mutual_info_classif(X_cadena, y)
    axis = X_cadena.columns
    ig_ordenados, axis_ordenadas = zip(*sorted(zip(ig, axis), reverse=True))
    feat_ig = pd.Series(ig_ordenados, axis_ordenadas)
    index = categories.index(category)
    fig, ax = plt.subplots(figsize=(size[index],11))
    feat_ig.plot.bar(color ='teal')
    ax.tick_params(axis='both', which='major', labelsize=5)
    #plt.show()
    fig.savefig("./img/information_gain/"+category+".png", dpi=300, bbox_inches='tight')

for feature in drop_features:
    X.drop(feature, axis= 1, inplace=True)
print('')

#---------------------------------
# # Correlation Analysis
#---------------------------------
print("Calculando analisis de correlacion ...")
# Create the correlation matrix using kendall coefficient
c_matrix = X.corr('kendall')

# Display and save a heatmap of the correlation matrix
fig = plt.figure(figsize=(11,11))
ax = fig.add_subplot()
sns.heatmap(c_matrix)
c_features = set()
for i in range(len(c_matrix.columns)):
    for j in range(i):
        if abs(c_matrix.iloc[i, j]) > 0.8:
            col = c_matrix.columns[i]
            c_features.add(col)

for feature in c_features:
    print("La caracteristica correlada {} va a ser eliminada".format(feature))
print('')

X.drop(labels=c_features, axis=1, inplace=True)

ax.tick_params(axis='both', which='major', labelsize=5)
#plt.show()
fig.savefig("./img/correlation_analysis/corr_matrix_NoDrop.png", dpi=300)

# Poteamos la gráfica después de haber eliminado las features correlacionadas
c_matrix = X.corr('kendall')
fig = plt.figure(figsize=(11,11))
ax = fig.add_subplot()
sns.heatmap(c_matrix)
ax.tick_params(axis='both', which='major', labelsize=5)
#plt.show()
fig.savefig("./img/correlation_analysis/corr_matrix_Drop.png", dpi=300)

#============================================================
# Eliminacion de Outliers
#============================================================

print("Eliminando outliers ...")

counter = Counter(y)
print(counter)
print([(i, counter[i] / sum(counter.values())  * 100.0) for i in counter])
print("Length before outliers dropped: {}".format(len(X)), "\n")

# Remove all rows where the specified column is +/- sd standard deviations

df = pd.concat([X,y], axis= 1)
df['outliers'] = 0

for i in range(12):
    df_i = df[df['Category'] == i]
    indices = df[df['Category'] == i].index
    for features in X.columns:
        mean = df_i[features].mean()
        std = df_i[features].std()
        condicion = ((df['Category'] == i) & (np.abs(df[features] - mean) >= (2 * std)))
        df.loc[condicion, 'outliers'] += 1

df = df.drop(df[df['outliers'] >2].index)

X = df.drop('outliers',axis= 1)
X = X.drop('Category',axis= 1)
y = df['Category']

counter = Counter(y)
print(counter)
print([(i, counter[i] / sum(counter.values())  * 100.0) for i in counter])
print("Length after outliers dropped: {}".format(len(X)), "\n")

print('Outliers eliminados\n')

#============================================================
# Balanceo del dataset
#============================================================

print('Balanceando el dataset ...')

df = pd.concat([X,y], axis= 1)
df_sup = df
df_inf = df

for i in range(12):
    df_i = df[df['Category'] == i]
    if len(df_i.index) > 4000:
        df_inf = df_inf[df_inf['Category'] != i]
    else :
        df_sup = df_sup[df_sup['Category'] != i]

X_sup = df_sup.drop('Category',axis= 1)
y_sup = df_sup['Category']
tl = TomekLinks()
X_sup, y_sup = tl.fit_resample(X_sup,y_sup)
enn = EditedNearestNeighbours()
X_sup, y_sup = enn.fit_resample(X_sup,y_sup)
# nm = NearMiss()
# X_sup, y_sup = nm.fit_resample(X_sup,y_sup)

X_inf = df_inf.drop('Category',axis= 1)
y_inf = df_inf['Category']

X = pd.concat([X_sup, X_inf], axis=0)
y = pd.concat([y_sup, y_inf], axis=0)

sm = SMOTE()
X, y = sm.fit_resample(X,y)

counter = Counter(y)
print(counter)
print([(i, counter[i] / sum(counter.values())  * 100.0) for i in counter])

print('Dataset Balanceado\n')

#============================================================
# Comprobamos que no hayan registros duplicados
#============================================================

print('Comprobando filas repetidas ...')

df = pd.concat([X,y], axis= 1)

df = df.drop_duplicates(df.columns[~df.columns.isin(['Hash'])], keep= 'first')

X = df.drop('Category',axis= 1)
y = df['Category']

print('Filas repetidas comprobadas\n')

#============================================================
# Dividimos en parte de Train y Test y guardamos
#============================================================

print('División en parte de Train y Test ...')
print(np.shape(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=41) #Elegir test entre 0,2 y 0,3 (máximo)
counter_y_train = Counter(y_train)
print('Train:')
print(counter_y_train)
print([(i, counter_y_train[i] / sum(counter_y_train.values())  * 100.0) for i in counter_y_train], "\n")
counter_y_test = Counter(y_test)
print('Test:')
print(counter_y_test)
print([(i, counter_y_test[i] / sum(counter_y_test.values())  * 100.0) for i in counter_y_test], "\n")

X_train.to_csv('./data/x_train.csv', index = False)
X_test.to_csv('./data/x_test.csv', index = False)
y_train.to_csv('./data/y_train.csv', index = False)
y_test.to_csv('./data/y_test.csv', index = False)

print('Data-set guardado correctamente', "\n")