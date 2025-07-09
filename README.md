# ❤️ Predicción de Enfermedad Cardíaca (Heart Disease Prediction)

## 📝 Descripción
Proyecto para predecir la presencia de una enfermedad cardíaca mediante clasificación binaria.

## 🛠️ Tecnologías y librerías
- 🐍 Python 3.13  
- 📊 pandas  
- 🔢 numpy  
- 📈 matplotlib  
- 🤖 scikit-learn 

## 📂 Dataset
Se usa el dataset *Heart Failure Prediction Dataset* https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
## 🗂️ Estructura del proyecto
````
heart-disease-prediction/
├── data/ # Dataset
├── heart_disease_part1.py # Código para carga y análisis inicial
├── requirements.txt # Dependencias
└── README.md # Este archivo
````
## ▶️ Cómo clonar y ejecutar el proyecto

1. Clonar el repositorio:

```bash
git clone https://github.com/leirevc/heart-disease-prediction
cd heart-disease-prediction

````
2.Crear y activar entorno virtual:
````
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
````
3. Instalar las dependencias
````
pip install -r requirements.txt
````
4.Ejecutar el script
````
python heart_disease_part1.py
````