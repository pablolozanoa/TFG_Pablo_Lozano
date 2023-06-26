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

## Ejecución
