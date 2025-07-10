import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    # Cargar y preparar datos
    df = pd.read_csv("../data/heart.csv")
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, drop_first=True)

    #Definir variables
    x = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # Dividir en train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Escalado
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    #Explorar diferentes k y elegir el mejor
    k_range = range(1, 21)
    accuracies = []
    for k in k_range:
        modelKNN = KNeighborsClassifier(n_neighbors=k)
        modelKNN.fit(x_train_scaled, y_train)
        y_pred = modelKNN.predict(x_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    best_k = k_range[np.argmax(accuracies)]
    print("\nKNN mejor K encontrado: ", best_k)

    #Modelo K Nearest Neighbors
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(x_train_scaled, y_train)
    y_pred = final_model.predict(x_test_scaled)

    #Predicción
    y_pred = modelKNN.predict(x_test_scaled)

    #Resultado
    print("\nResultados - K-Nearest Neighbors(KNN)")
    print(f"Precisión: {accuracy_score(y_test, y_pred):.3f}%") #Ajuste a 3 decimales
    print(f"Matriz de confusión:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Reporte de clasificación:\n{classification_report(y_test, y_pred)}")

    # Representar la matriz de confusión de forma gráfica
    confmatrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(confmatrix, annot=True, fmt="d", cmap='Blues', cbar=False)
    plt.xlabel('Preddición')
    plt.ylabel('Actual')
    plt.title('Matriz de confusión de K-Nearest Neighbors(KNN)')
    plt.show()

    #Gráfico de precisión/números de vecinos
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, accuracies, marker='o', linestyle='-', markerfacecolor='red', markersize=5)
    plt.xlabel('Número de vecinos (k)')
    plt.ylabel('Predicción')
    plt.title('Precisión vs Número de vecinos (k)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    main()