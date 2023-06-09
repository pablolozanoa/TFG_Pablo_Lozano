import sys
import joblib
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from kafka                          import KafkaConsumer
from sklearn.discriminant_analysis  import StandardScaler
from tabulate                       import tabulate



bootstrap_servers = ['localhost:9092']
topicName = 'Attacks'
print('Creando el consumidor ...')
consumer = KafkaConsumer (topicName, group_id = 'group1',bootstrap_servers = bootstrap_servers, auto_offset_reset = 'earliest')
print('Consumidor creado\n')

print('Cargando los modelos ...')
rf = joblib.load('./saved_model/rf_model.pkl')
svm = joblib.load('./saved_model/svm_model.pkl')
knn = joblib.load('./saved_model/knn_model.pkl')
lstm = tf.keras.models.load_model('./saved_model/lstm_model.h5')
print('Modelos cargados')

recibido = []

def predict_GC(df_muestra):

    rf_pred = rf.predict_proba(df_muestra)
    rf_res_tot = rf.predict(df_muestra)
    scaler = StandardScaler().fit(df_muestra)
    X_test_svm = scaler.transform(df_muestra)
    svm_pred = svm.predict_proba(X_test_svm)
    svm_res_tot = svm.predict(X_test_svm)
    knn_pred = knn.predict_proba(df_muestra)
    knn_res_tot = knn.predict(df_muestra)
    X_test_lstm = np.reshape(df_muestra.values, (df_muestra.shape[0], 1, df_muestra.shape[1]))
    lstm_pred = lstm.predict(X_test_lstm, verbose = 0)
    lstm_res_tot= []
    for i in range(len(lstm_pred)):
        lstm_res_tot.append(np.argmax(lstm_pred[i]))
    
    tprs = {
        'KNN':[0.767, 0.988, 1.0, 0.988, 0.875, 0.930, 0.997, 0.959, 1.0, 0.959, 0.965, 0.953],
        'SVM': [0.923, 0.979, 1.0, 0.991, 0.822, 0.940, 1.0, 0.963, 0.999, 0.828, 0.973,0.952],
        'RF': [0.934, 0.964, 0.998, 0.974, 0.887, 0.935, 0.999, 0.934, 0.999, 0.822, 0.919, 0.915],
        'LSTM': [0.925, 0.967, 1.0, 0.994, 0.818, 0.931, 0.998, 0.957, 0.999, 0.848, 0.945, 0.953],
    }
    df_ponderaciones = pd.DataFrame(columns=['RF_pond', 'SVM_pond', 'KNN_pond', 'LSTM_pond'])
    for i in range(len(tprs['RF'])):
        pot = 10
        tot = (pow(tprs['RF'][i], pot) + pow(tprs['SVM'][i], pot) + pow(tprs['KNN'][i], pot)+ pow(tprs['LSTM'][i], pot))
        rf_pond = pow(tprs['RF'][i], pot) / tot
        svm_pond = pow(tprs['SVM'][i], pot) / tot
        knn_pond = pow(tprs['KNN'][i], pot) / tot
        lstm_pond = pow(tprs['LSTM'][i], pot) / tot
        df_ponderaciones.loc[i] = [rf_pond, svm_pond, knn_pond, lstm_pond]

    rf_pred_pond = np.zeros_like(rf_pred)
    svm_pred_pond = np.zeros_like(svm_pred)
    knn_pred_pond = np.zeros_like(knn_pred)
    lstm_pred_pond = np.zeros_like(lstm_pred)

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

    type_argmax= []
    for i in range(len(df_muestra)):
        type_argmax.append(np.argmax(pred_total[i]))
    
    recibido.append(type_argmax[0])

    aux.insert(0,categorias[type_argmax[0]])
    fecha_hora_actual = datetime.datetime.now()
    fecha_hora_actual_str = fecha_hora_actual.strftime("%Y-%m-%d %H:%M:%S")
    aux.insert(0,fecha_hora_actual_str)

    nuevo_indice = logs.shape[0] +1
    logs.loc[nuevo_indice] = aux

    return type_argmax[0]

def guardar_logs():
    fecha_hora_actual = datetime.datetime.now()
    fecha_hora_actual_str = fecha_hora_actual.strftime("%Y-%m-%d_%H-%M-%S")
    logs.to_csv('./real_time_logs/logs_'+ fecha_hora_actual_str +'.csv')

    print('Se han almacenado los logs en el archivo logs_'+ fecha_hora_actual_str +'\n')

def guardar_resultados():
    etiquetas = [categorias[i] for i in range(12)]
    plt.xticks(x_values, etiquetas, rotation=45, ha = 'right')
    plt.xlabel('Nº Muestras', fontsize=8)
    plt.ylabel('Categorías', fontsize=8)
    plt.title('Histograma de los ataques recibidos', fontsize=15, pad =10, fontname='Times New Roman')
    plt.yticks(np.arange(0, max(count) + 2, 1))
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    plt.savefig("./img/RT/Real_Time.png", dpi=300)

    print('Se ha guardado el Histograma de los ataques recibidos\n')

c_matrix = np.zeros((12, 12))
axis = 0
categorias = {0 :  "Adware", 1 :  "Backdoor", 2 :  "FileInfector", 3 :  "PUA", 4 :  "Ransomware", 5 :  "Riskware", 6 :  "Scareware", 7 :  "Trojan", 8 :  "Trojan_Banker", 9 :  "Trojan_Dropper", 10: "Trojan_SMS", 11: "Trojan_Spy"}
logs = pd.DataFrame(columns=['Fecha', 'Predicción', 'Memory_PssTotal', 'Memory_PssClean', 'Memory_SharedDirty', 'Memory_PrivateDirty', 'Memory_SharedClean', 'Memory_HeapSize', 'Memory_HeapFree', 'Memory_Views', 'Memory_ViewRootImpl', 'Memory_AppContexts', 'Memory_Activities', 'Memory_Assets', 'Memory_LocalBinders', 'Memory_ProxyBinders', 'Memory_ParcelMemory', 'Memory_ParcelCount', 'Memory_DeathRecipients', 'Memory_WebViews', 'API_Command_java.lang.Runtime_exec', 'API_WebView_android.webkit.WebView_loadUrl', 'API_WebView_android.webkit.WebView_addJavascriptInterface', 'API_FileIO_libcore.io.IoBridge_open', 'API_FileIO_android.content.ContextWrapper_openFileInput', 'API_FileIO_android.content.ContextWrapper_openFileOutput', 'API_FileIO_android.content.ContextWrapper_deleteFile', 'API_Database_android.database.sqlite.SQLiteDatabase_execSQL', 'API_Database_android.database.sqlite.SQLiteDatabase_getPath', 'API_Database_android.database.sqlite.SQLiteDatabase_insert', 'API_Database_android.database.sqlite.SQLiteDatabase_query', 'API_Database_android.database.sqlite.SQLiteDatabase_rawQuery', 'API_Database_android.database.sqlite.SQLiteDatabase_update', 'API_IPC_android.content.ContextWrapper_sendBroadcast', 'API_IPC_android.content.ContextWrapper_startActivity', 'API_IPC_android.content.ContextWrapper_startService', 'API_IPC_android.content.ContextWrapper_stopService', 'API_IPC_android.content.ContextWrapper_registerReceiver', 'API_Binder_android.app.ActivityThread_handleReceiver', 'API_Binder_android.app.Activity_startActivity', 'API_Crypto_javax.crypto.spec.SecretKeySpec_$init', 'API_Crypto-Hash_java.security.MessageDigest_digest', 'API_DeviceInfo_android.telephony.TelephonyManager_getDeviceId', 'API_DeviceInfo_android.telephony.TelephonyManager_getSubscriberId', 'API_DeviceInfo_android.telephony.TelephonyManager_getLine1Number', 'API_DeviceInfo_android.telephony.TelephonyManager_getNetworkOperator', 'API_DeviceInfo_android.telephony.TelephonyManager_getNetworkOperatorName', 'API_DeviceInfo_android.net.wifi.WifiInfo_getMacAddress', 'API_DeviceInfo_android.telephony.TelephonyManager_getSimSerialNumber', 'API_DeviceInfo_android.telephony.TelephonyManager_getNetworkCountryIso', 'API_Network_java.net.URL_openConnection', 'API_Network_com.android.okhttp.internal.huc.HttpURLConnectionImpl_getInputStream', 'API_DexClassLoader_dalvik.system.BaseDexClassLoader_findResource', 'API_DexClassLoader_dalvik.system.BaseDexClassLoader_findLibrary', 'API_DexClassLoader_dalvik.system.DexClassLoader_$init', 'API_Base64_android.util.Base64_decode', 'API_Base64_android.util.Base64_encode', 'API_SystemManager_android.app.ApplicationPackageManager_setComponentEnabledSetting', 'API_SystemManager_android.content.BroadcastReceiver_abortBroadcast', 'API_DeviceData_android.content.ContentResolver_query', 'API_DeviceData_android.content.ContentResolver_registerContentObserver', 'API_DeviceData_android.os.SystemProperties_get', 'API_DeviceData_android.app.ApplicationPackageManager_getInstalledPackages', 'API__sessions', 'Network_TotalReceivedBytes', 'Battery_wakelock', 'Battery_service', 'Hash'])

try:

    for message in consumer:
        aux = message.value.decode('utf-8').rstrip('\n').split(',')
        aux = list(map(float, aux))
        print(aux)
        print(type(aux))
        df = pd.DataFrame(columns=['Memory_PssTotal', 'Memory_PssClean', 'Memory_SharedDirty', 'Memory_PrivateDirty', 'Memory_SharedClean', 'Memory_HeapSize', 'Memory_HeapFree', 'Memory_Views', 'Memory_ViewRootImpl', 'Memory_AppContexts', 'Memory_Activities', 'Memory_Assets', 'Memory_LocalBinders', 'Memory_ProxyBinders', 'Memory_ParcelMemory', 'Memory_ParcelCount', 'Memory_DeathRecipients', 'Memory_WebViews', 'API_Command_java.lang.Runtime_exec', 'API_WebView_android.webkit.WebView_loadUrl', 'API_WebView_android.webkit.WebView_addJavascriptInterface', 'API_FileIO_libcore.io.IoBridge_open', 'API_FileIO_android.content.ContextWrapper_openFileInput', 'API_FileIO_android.content.ContextWrapper_openFileOutput', 'API_FileIO_android.content.ContextWrapper_deleteFile', 'API_Database_android.database.sqlite.SQLiteDatabase_execSQL', 'API_Database_android.database.sqlite.SQLiteDatabase_getPath', 'API_Database_android.database.sqlite.SQLiteDatabase_insert', 'API_Database_android.database.sqlite.SQLiteDatabase_query', 'API_Database_android.database.sqlite.SQLiteDatabase_rawQuery', 'API_Database_android.database.sqlite.SQLiteDatabase_update', 'API_IPC_android.content.ContextWrapper_sendBroadcast', 'API_IPC_android.content.ContextWrapper_startActivity', 'API_IPC_android.content.ContextWrapper_startService', 'API_IPC_android.content.ContextWrapper_stopService', 'API_IPC_android.content.ContextWrapper_registerReceiver', 'API_Binder_android.app.ActivityThread_handleReceiver', 'API_Binder_android.app.Activity_startActivity', 'API_Crypto_javax.crypto.spec.SecretKeySpec_$init', 'API_Crypto-Hash_java.security.MessageDigest_digest', 'API_DeviceInfo_android.telephony.TelephonyManager_getDeviceId', 'API_DeviceInfo_android.telephony.TelephonyManager_getSubscriberId', 'API_DeviceInfo_android.telephony.TelephonyManager_getLine1Number', 'API_DeviceInfo_android.telephony.TelephonyManager_getNetworkOperator', 'API_DeviceInfo_android.telephony.TelephonyManager_getNetworkOperatorName', 'API_DeviceInfo_android.net.wifi.WifiInfo_getMacAddress', 'API_DeviceInfo_android.telephony.TelephonyManager_getSimSerialNumber', 'API_DeviceInfo_android.telephony.TelephonyManager_getNetworkCountryIso', 'API_Network_java.net.URL_openConnection', 'API_Network_com.android.okhttp.internal.huc.HttpURLConnectionImpl_getInputStream', 'API_DexClassLoader_dalvik.system.BaseDexClassLoader_findResource', 'API_DexClassLoader_dalvik.system.BaseDexClassLoader_findLibrary', 'API_DexClassLoader_dalvik.system.DexClassLoader_$init', 'API_Base64_android.util.Base64_decode', 'API_Base64_android.util.Base64_encode', 'API_SystemManager_android.app.ApplicationPackageManager_setComponentEnabledSetting', 'API_SystemManager_android.content.BroadcastReceiver_abortBroadcast', 'API_DeviceData_android.content.ContentResolver_query', 'API_DeviceData_android.content.ContentResolver_registerContentObserver', 'API_DeviceData_android.os.SystemProperties_get', 'API_DeviceData_android.app.ApplicationPackageManager_getInstalledPackages', 'API__sessions', 'Network_TotalReceivedBytes', 'Battery_wakelock', 'Battery_service', 'Hash'])
        df.loc[0] = aux

        # Llamar una función para detectar la categoría y luego imprimir por pantalla
        prediccion = predict_GC(df)
        print('Se ha detectado un ataque de tipo {}\n'.format(categorias[prediccion]))
        print('El recuento total de ataques detectados es:')

        resultados = []
        for x in range(0, 12):
            count = recibido.count(x)
            resultados.append([categorias[x], count])
        print(tabulate(resultados, headers=['Categoría del ataque', 'Nº ataques']))
        print('')

        # Ploteamos el histograma de los datos que estamos recibiendo
        plt.pause(0.002)
        plt.clf()
        count = np.bincount(recibido, minlength=12)
        x_values = np.arange(0, 12)
        plt.bar(x_values, count, align='center', alpha=0.5, color='teal')
        etiquetas = [categorias[i] for i in range(12)]
        plt.xticks(x_values, etiquetas, rotation=45, ha = 'right')
        plt.xlabel('Nº Muestras', fontsize=8)
        plt.ylabel('Categorías', fontsize=8)
        plt.title('Histograma de los ataques recibidos', fontsize=15, pad =10, fontname='Times New Roman')
        plt.yticks(np.arange(0, max(count) + 2, 1))
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.tight_layout()

        plt.pause(0.002)
        plt.pause(0.001)

except KeyboardInterrupt:
    print('\nSe ha cerrado la conexión\n')
    print('Se va a guardar la información recibida ...\n')
    guardar_resultados()
    guardar_logs()
    print('\nTodo ha sido almacenado correctamente')
    sys.exit()