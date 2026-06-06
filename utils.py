"""Funciones utilitarias para evaluación y estadísticas del dataset.
Contiene evaluación del modelo y funciones estadísticas solicitadas.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(dataset_path: str = "dataset.csv", model_path: str = "model.pkl", scaler_path: str = "scaler.pkl") -> None:
    """Evalúa el modelo entrenado sobre el conjunto de prueba y muestra métricas.

    Métricas mostradas: Accuracy, Precision, Recall, F1, Matriz de Confusión
    """
    try:
        # Verificaciones de existencia de archivos
        if not os.path.exists(dataset_path):
            print(f"No se encontró {dataset_path}. Genere el dataset primero.")
            return

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("No se encontró el modelo o el scaler. Entrene el modelo primero.")
            return

        # Cargar datos y separar X/y
        df = pd.read_csv(dataset_path)
        X = df[["CPU", "RAM", "Temperatura", "Red"]]
        y = df["Fallo"]

        # Dividir en 70/30 reproducible
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Cargar scaler y modelo
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Escalar conjunto de prueba
        X_test_scaled = scaler.transform(X_test)

        # Predicciones y probabilidades
        y_pred = model.predict(X_test_scaled)

        # Cálculo de métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Mostrar métricas en formato legible
        print("\nEvaluación del modelo:\n")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}\n")
        print("Matriz de Confusión:")
        print(cm)
        print("")

    except Exception as e:
        print(f"Error durante la evaluación: {e}")


def show_statistics(dataset_path: str = "dataset.csv") -> None:
    """Muestra estadísticas básicas del dataset: cantidad, columnas, promedio, min, max, std y distribución de Fallo."""
    try:
        # Verificar existencia del dataset
        if not os.path.exists(dataset_path):
            print(f"No se encontró {dataset_path}. Genere el dataset primero.")
            return

        # Cargar dataset
        df = pd.read_csv(dataset_path)

        # Cantidad de registros y columnas
        n_rows = df.shape[0]
        n_cols = df.shape[1]

        print("\nEstadísticas del dataset:\n")
        print(f"Cantidad de registros: {n_rows}")
        print(f"Cantidad de columnas : {n_cols}\n")

        # Estadísticas resumen para las columnas numéricas
        desc = df.describe()
        # Mostrar promedio, min, max, std por columna en formato legible
        for col in ["CPU", "RAM", "Temperatura", "Red"]:
            mean = desc.loc["mean", col]
            mn = desc.loc["min", col]
            mx = desc.loc["max", col]
            std = desc.loc["std", col]
            print(f"{col}: promedio={mean:.2f}, min={mn:.2f}, max={mx:.2f}, std={std:.2f}")

        # Distribución de la variable Fallo
        dist = df["Fallo"].value_counts().sort_index()
        total = n_rows
        print("\nDistribución de 'Fallo':")
        for val, cnt in dist.items():
            pct = (cnt / total) * 100.0
            print(f"Fallo={val}: {cnt} registros ({pct:.2f}%)")

        print("")

    except Exception as e:
        print(f"Error al mostrar estadísticas: {e}")


if __name__ == "__main__":
    # Permite pruebas rápidas desde la línea de comandos
    show_statistics()
