import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    #Carga del dataset
    df = pd.read_csv("data/heart.csv")
    #Exploración de los datos básicos
    print("Primeras 5 filas del dataset:\n", df.head())
    print("\nDimensiones del dataset:\n", df.shape)
    print("\nTipos de datos de las columnas:\n", df.dtypes)

    #Distribución de la variable objetivo-target
    print("\nDistribución de la variable target:\n", df['HeartDisease'].value_counts())
    #Representar el target
    df['HeartDisease'].value_counts().plot(kind='bar')
    plt.title("Histograma de la variable target")
    plt.xlabel("Presencia de enfermedad (1 = sí, 0 = no")
    plt.ylabel("Cantidad de casos")
    plt.show()

    #Verificar y limpiar valores nulos
    print("\nValores nulos por columna:\n", df.isnull().sum())
    if df.isnull().sum().sum() > 0:
        print("\nExisten valores nulos, se eliminarán las filas que los contienen:\n")
        df =  df.dropna()
        print("Nuevo tamaño del dataset: {df.shape}")

    #Converti columnas categóricas
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        print(f"\nColumnas categóricas encontradas: {list(cat_cols)}")
        original_cols = set(df.columns)

        df = pd.get_dummies(df, drop_first=True)
        new_cols = set(df.columns) - original_cols
        print("Después de la conversión con get_dummies:")
        print("Nuevas columnas creadas:", list(new_cols))
        print(df.head())
    else:
        print("\nNo se encontraron columnas categóricas.")



if __name__ == "__main__":
    main()