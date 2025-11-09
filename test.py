import pandas as pd
df = pd.read_csv("housing.csv")
print(df['ocean_proximity'].unique())

# Archivo utilizado para realizar análisis estadiísticos sobre el dataset.
