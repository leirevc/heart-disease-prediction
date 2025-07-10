# Predicción de Enfermedad Cardíaca (Heart Disease Prediction)

## 📝 Descripción
Proyecto de exploración y comparativa entre la predicción de diferentes modelos sobre un mismo Dataset.

---

## 🛠️ Tecnologías y librerías
- 🐍 Python
- 📊 pandas  
- 🔢 numpy  
- 📈 matplotlib
- 🌊 seaborn  
- 🤖 scikit-learn 

---

## 📂 Dataset
Se usa el dataset *Heart Failure Prediction Dataset* https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

---

## 🗂️ Estructura del proyecto
````
heart-disease-prediction/
├── data/ # Dataset
├── Models/
│   ├── logistic_regression.py  # Modelo de Regresión Logística
│   ├── knn.py                  # Modelo K-Nearest Neighbors (KNN)
│   └── ...                     # Otros modelos futuros
├── heart_disease_part1.py # Código para carga y análisis inicial
├── requirements.txt # Dependencias
└── README.md # Este archivo
````

---

## ⚙️ Cómo ejecutar

1. Clona este repositorio  
2. Crea y activa un entorno virtual en Python  
3. Instala las dependencias con `pip install -r requirements.txt`  
4. Ejecuta los scripts para ver el resultado de cada modelo


---
## 🤖 Modelos utilizados
| Modelo                  | Accuracy | Observaciones                                                                       |
|-------------------------|----------|-------------------------------------------------------------------------------------|
| 🔹 Regresión Logística  | 0.853    | Buen rendimiento general, rápido de entrenar                                        |
| 🔹 K-Nearest Neighbors (KNN) | ~0.85   | Mejora ajustando el valor de *k*, más sensible al escalado. Mejor resultado *k* = 7 |



---

## 🙋 Sobre mí
Soy ingeniera de sistemas de información con muchas ganas de aprender, crecer y experimentar para orientar mi carrera hacia Data e IA 🚀.