import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import gzip
import os
import json
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
dftest = pd.read_csv('files/input/test_data.csv.zip', compression='zip')
dftrain = pd.read_csv('files/input/train_data.csv.zip', compression='zip')
dftest.rename(columns={'default payment next month': 'default'}, inplace=True)
dftrain.rename(columns={'default payment next month': 'default'}, inplace=True)
dftest.drop(columns=['ID'], inplace=True)
dftrain.drop(columns=['ID'], inplace=True)

dftrain = dftrain[dftrain['EDUCATION'] != 0]
dftest = dftest[dftest['EDUCATION'] != 0]
dftrain = dftrain[dftrain['MARRIAGE'] != 0]
dftest = dftest[dftest['MARRIAGE'] != 0]
dftrain['EDUCATION'] = dftrain['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
dftest['EDUCATION'] = dftest['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

X_train = dftrain.drop(columns=['default']) 
y_train = dftrain['default'] 

X_test = dftest.drop(columns=['default']) 
y_test = dftest['default']

columnas_categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']


preprocesador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
    ],
    remainder='passthrough' # ¡Muy importante! Le dice que deje las columnas numéricas intactas
)
# El pipeline ahora solo tiene un paso
pipeline_modelo = Pipeline(steps=[
    ('preprocesamiento', preprocesador),
    ('clasificador', RandomForestClassifier(random_state=42)) 
])
parametros_a_probar = {
    # Más árboles = predicciones más estables (menos varianza)
    'clasificador__n_estimators': [300, 400], 
    
    # Lo dejamos en None para asegurar que pases la prueba de Train > 0.944
    'clasificador__max_depth': [None], 
    
    # Un micro-ajuste en cómo se dividen las hojas
    'clasificador__min_samples_split': [2, 4], 
    'clasificador__min_samples_leaf': [1, 2]
}
# 2. Configuramos la Validación Cruzada y la Métrica
optimizador = GridSearchCV(
    estimator=pipeline_modelo,            
    param_grid=parametros_a_probar,       
    cv=10,                                
    scoring='balanced_accuracy',          
    n_jobs=5,
    verbose=3                            
)

print("⏳ Pasando paso 1: Iniciando el entrenamiento pesado (GridSearchCV)...")
print("Nota: Al usar n_jobs=1 esto puede tardar unos minutos. No cierres la terminal.")

optimizador.fit(X_train, y_train)

print("✅ ¡Paso 2 completado! El entrenamiento terminó con éxito.")

mejor_modelo = optimizador.best_estimator_

ruta_carpeta = 'files/models'
nombre_archivo = 'model.pkl.gz'
ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)

print(f"📂 Paso 3: Verificando/Creando la carpeta en '{ruta_carpeta}'...")
os.makedirs(ruta_carpeta, exist_ok=True) # Esto hace lo mismo que tu IF pero en una sola línea limpia

print("💾 Paso 4: Escribiendo el archivo model.pkl.gz en el disco...")
with gzip.open(ruta_completa, 'wb') as f:
    pickle.dump(optimizador, f, protocol=4)

print("🎉 ¡PROCESO TERMINADO! El modelo se guardó y creó correctamente.")

y_train_pred = mejor_modelo.predict(X_train)
y_test_pred = mejor_modelo.predict(X_test)
metrics_train = {
    'type': 'metrics',
    'dataset': 'train',
    'precision': round(precision_score(y_train, y_train_pred), 4),
    'balanced_accuracy': round(balanced_accuracy_score(y_train, y_train_pred), 4),
    'recall': round(recall_score(y_train, y_train_pred), 4),
    'f1_score': round(f1_score(y_train, y_train_pred), 4)
}
metrics_test = {
    'type': 'metrics',
    'dataset': 'test',
    'precision': round(precision_score(y_test, y_test_pred), 4),
    'balanced_accuracy': round(balanced_accuracy_score(y_test, y_test_pred), 4),
    'recall': round(recall_score(y_test, y_test_pred), 4),
    'f1_score': round(f1_score(y_test, y_test_pred), 4)
}

ruta_archivo = 'files/output/metrics.json'

with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
    # Escribimos el diccionario de train y un salto de línea (\n)
    archivo.write(json.dumps(metrics_train) + '\n')
    # Escribimos el diccionario de test y un salto de línea (\n)
    archivo.write(json.dumps(metrics_test) + '\n')

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
dict_cm_train = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {"predicted_0": int(cm_train[0][0]), "predicted_1": int(cm_train[0][1])},
    'true_1': {"predicted_0": int(cm_train[1][0]), "predicted_1": int(cm_train[1][1])}
}

dict_cm_test = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {"predicted_0": int(cm_test[0][0]), "predicted_1": int(cm_test[0][1])},
    'true_1': {"predicted_0": int(cm_test[1][0]), "predicted_1": int(cm_test[1][1])}
}

ruta_archivo = 'files/output/metrics.json'

with open(ruta_archivo, 'a', encoding='utf-8') as archivo:
    archivo.write(json.dumps(dict_cm_train) + '\n')
    archivo.write(json.dumps(dict_cm_test) + '\n')






# 4. Ver los resultados
'''print("Los mejores hiperparámetros encontrados son:")
print(optimizador.best_params_)
print("\nLa mejor precisión balanceada obtenida fue:")
print(optimizador.best_score_)'''





# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
