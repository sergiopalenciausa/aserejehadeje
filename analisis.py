import pandas as pd
data_trabajo= pd.read_excel('data_completa.xlsx')
print(data_trabajo.head())

print(data_trabajo.describe())