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



import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Suponiendo que 'datos' es tu DataFrame con valores

# Normaliza los datos
datos_normalizados = (datos_imputados_df - datos_imputados_df.min()) / (datos_imputados_df.max() - datos_imputados_df.min())

# Define la arquitectura del Autoencoder
input_dim = len(datos_imputados_df.columns)
encoding_dim = 32  # Dimensión de la capa de codificación

input_data = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_data)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Entrena el Autoencoder
autoencoder.fit(datos_normalizados, datos_normalizados, epochs=50, batch_size=32, shuffle=True)

# Genera nuevos datos sintéticos
datos_generados_normalizados = autoencoder.predict(datos_normalizados)

# Convierte los datos generados a la escala original
datos_generados = datos_generados_normalizados * (datos_imputados_df.max() - datos_imputados_df.min()) + datos_imputados_df.min()

# Convierte los datos generados a un DataFrame
datos_generados_df = pd.DataFrame(datos_generados, columns=datos_imputados_df.columns)

print(datos_imputados_df.head())
