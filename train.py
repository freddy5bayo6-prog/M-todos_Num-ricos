"""Módulo para entrenar el modelo de Regresión Logística.
Carga `dataset.csv`, divide datos, normaliza y entrena el modelo.
Guarda `model.pkl` y `scaler.pkl`.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def train_model(dataset_path: str = "dataset.csv", model_path: str = "model.pkl", scaler_path: str = "scaler.pkl") -> None:
    """Entrena un modelo de regresión logística y guarda el modelo y el scaler.

    Pasos:
    - Cargar dataset
    - Dividir 70/30
    - Normalizar variables
    - Entrenar LogisticRegression
    - Guardar model.pkl y scaler.pkl
    """
    try:
        # Verificar existencia del dataset
        if not os.path.exists(dataset_path):
            print(f"No se encontró {dataset_path}. Genere el dataset primero.")
            return

        # Cargar CSV en DataFrame
        df = pd.read_csv(dataset_path)

        # Separar características (X) y etiqueta (y)
        X = df[["CPU", "RAM", "Temperatura", "Red"]]
        y = df["Fallo"]

        # Dividir datos: 70% entrenamiento, 30% prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Normalizar usando StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Definir y entrenar el modelo de regresión logística
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Guardar el modelo y el scaler en disco mediante pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Mensaje de confirmación
        print("Modelo entrenado correctamente.")

    except Exception as e:
        # Manejo de errores para informar al usuario
        print(f"Error durante el entrenamiento: {e}")


if __name__ == "__main__":
    # Permite ejecutar directamente para pruebas
    train_model()
