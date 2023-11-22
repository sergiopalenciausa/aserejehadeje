import pandas as pd

# Lee un archivo Excel
datos_excel = pd.read_excel('aguavista.xlsx')

# Muestra los primeros registros para verificar la importación
print(datos_excel.head())


columnas_a_mantener = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Turbidity', 'Potability']

# Selecciona las columnas deseadas para crear un nuevo DataFrame
nuevos_datos = datos_excel[columnas_a_mantener]
print(nuevos_datos.head())


# Suponiendo que 'datos' es tu DataFrame
# Utiliza el método isnull() para identificar los valores nulos y sum() para contarlos por columna
faltantes_por_columna = nuevos_datos.isnull().sum()

# Muestra el recuento de valores nulos por columna
print(faltantes_por_columna)



import pandas as pd
from sklearn.impute import KNNImputer

# Suponiendo que 'datos' es tu DataFrame con valores faltantes

# Crea un objeto KNNImputer con el número de vecinos deseado
imputador_knn = KNNImputer(n_neighbors=5)

# Imputa los valores faltantes en 'datos'
datos_imputados = imputador_knn.fit_transform(nuevos_datos)

# Convierte los datos imputados a un nuevo DataFrame (si es necesario)
datos_imputados_df = pd.DataFrame(datos_imputados, columns=nuevos_datos.columns)

print(datos_imputados_df.head())

numero_registros = datos_imputados_df.shape[0]
print("Número de registros:", numero_registros)

numero_muestras_generadas = 16724 # Cambia este valor según la cantidad deseada de datos adicionales

# Realiza bootstrapping para generar datos adicionales
datos_generados = []
for _ in range(numero_muestras_generadas):
    # Realiza muestreo con reemplazo sobre los datos originales
    muestra_bootstrap = datos_imputados_df.sample(n=len(datos_imputados_df), replace=True)
    
    # Agrega la muestra al conjunto de datos generados
    datos_generados.append(muestra_bootstrap)

# Crea un nuevo DataFrame con los datos generados
datos_generados_df = pd.concat(datos_generados, ignore_index=True)

print(datos_generados_df.head())
print(datos_generados_df.shape)

data_completa = datos_generados_df.head(20000)
print(data_completa.shape)

data_terminada = "data_completa.xlsx"

data_completa.to_excel(data_terminada, index = False )
