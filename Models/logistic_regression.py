import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

def main():
   #Cargar y preparar datos
    df = pd.read_csv("../data/heart.csv")
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, drop_first=True)

    #Definir variables
    x = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]


   #Dividir en train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

   #Escalado
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

   #Modelo de regresión logística
    modelLogisticRegression = LogisticRegression()
    modelLogisticRegression.fit(x_train_scaled, y_train)
    y_pred = modelLogisticRegression.predict(x_test_scaled)
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

   #Resultado
    print("\nResultados - Regresión Logística")
    print(f"Precisión: {accuracy_score(y_test, y_pred):.3f}") #Ajuste a 3 decimales
    print(f"Matriz de confusión:\n{confusion_matrix(y_test, y_pred)}\n")
    print(f"Reporte de clasificación:\n{classification_report(y_test, y_pred)}\n")

   #Representar la matriz de confusión de forma gráfica
    confmatrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (6,5))
    sns.heatmap(confmatrix, annot=True, fmt="d", cmap='Blues', cbar=False)
    plt.xlabel('Preddición')
    plt.ylabel('Actual')
    plt.title('Matriz de confusión del modelo de regresión logística')
    plt.show()

   #Representar la curva ROC de forma gráfica
    plt.figure(figsize = (6,5))
    plt.plot(fpr, tpr, color = 'red', lw=2, label=f'ROC curve (AUC = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC - Regresión logística')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()