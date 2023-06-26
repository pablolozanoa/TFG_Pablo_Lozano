# TFG_Pablo_Lozano
 IDS Model for Android Malware Classification

## Requisitos
Para poder ejecutar el código es necesario disponer de lo siguiente:
- Anaconda3
- Python
- Kafka y Zookeeper
- Librerías de Python:
    - TensorFlow
    - Keras
    - Scikit-Learn
    - Pandas
    - Matplotlib
    - Joblib
    - Seaborn
    - Bayes_opt
    - Kafka-python

## Instalación
Para evitar incompatibilidades de algunas librerías con la versión de Python, este proyecto fue desarrollado en un entorno de Anaconda3. Para ello hay que crear el entorno de la siguiente manera:
```
conda create --name nombre_del_entorno python=numero_versión
```
Siendo:
- _**nombre_del_entorno**_ un nombre a elegir
- _**numero_verión**_ la versión que se desea tener el en entorno

Y una vez creado activarlo:
```
conda activate nombre_del_entorno
```

Una vez se tiene el entorno, se instalan todas las librerías indicadas en la parte de los requisitos de la siguiente forma:
```
pip install nombre_de_la_librería
```
Siendo _**nombre_de_la_librería**_ la librería que se desea instalar

También hay que descargar e instalar la versión de [Python](https://www.python.org/downloads/) que se desee desde la página oficial

Por último, es necesario instalar [Kafka y Zookeper](https://kafka.apache.org/downloads) descargándolo desde la página oficial. Una vez descargado habrá que modificar los archivos de configuración “server” y “zookeeper” para indicar la ruta donde se ha instalado.

## Estructura
El repositorio está formado de 10 scripts y 6 carpetas. Hay un script encargado de ejecutar el preprocesado, otros 5 encargados de ejecutar los modelos individuales, otro que contiene las funciones necesarias para los modelos, otro para ejecutar el clasificador global y dos últimos para el sistema en tiempo real. Por otro lado, una de las carpetas contiene los data-sets originales y el resto de las carpetas sirven para almacenar la información y los gráficos extraída de los scripts.
## Ejecución
El primer paso es ejecutar el script encargado de realizar el preprocesado, para ello habrá que ejecutar el siguiente comando:
```
python preprocesado.py
```
Tras esta ejecución se generan en la carpeta “data” 4 archivos CSV. Dos con los datos de los data-sets tanto la parte de train como la de test y otros dos con sus respectivas características reales.

Después se esto, se ejecutarán los 5 modelos individuales:
```
python random_forest.py
```
```
python svm.py
```
```
python knn.py
```
```
python mlp.py
```
```
python lstm.py
```
Con estas ejecuciones se guardan en la carpeta “img” la matriz de confusión y la curva ROC de cada uno de los modelos. También en la carpeta “analysis” se guarda un CSV con la información detallada de las predicciones de cada muestra. Finalmente, en la carpeta “saved_model” se guardará el modelo entrenado.

Para ejecutar el clasificador global hay que ejecutar el siguiente comando:
```
python clasificador_global.py
```
Por último, para ejecutar el sistema en tiempo real, hay que desplegar el Broker de Kafka. Para ello habrá que abrir 4 terminales que harán lo siguiente:

- **Terminal 1:** En este se iniciará el servidor de Zookeeper. Para ello hay que ejecutar el siguiente comando desde la ruta \bin\windows de la carpeta donde se instaló Kafka. Este comando sirve solo para SO Windows.
```
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
```
- **Terminal 2:** En este se iniciará el servidor de Kafka. Para ello hay que ejecutar el siguiente comando desde la misma ruta que el Terminal 1. Este comando sirve solo para SO Windows.
```
.\bin\windows\kafka-server-start.bat .\config\server.properties
```
- **Terminal 3:** En este se ejecutará el productor de Kafka.
```
python producer.py
```
- **Terminal 4:** En este se ejecutará el consumidor de Kafka.
```
python consumer.py
```
Con estos 4 terminales se tendrá desplegado el sistema en tiempo real que tras su ejecución generará en las carpetas “img/RT” y “real_time_logs” los resultados obtenidos.
