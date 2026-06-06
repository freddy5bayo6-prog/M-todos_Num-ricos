"""Módulo para realizar predicciones manuales desde la consola.
Pide al usuario los valores de CPU, RAM, Temperatura y Red,
los normaliza con el scaler guardado y usa el modelo para predecir.
"""

import os
import pickle
import numpy as np


def _read_int(prompt: str, min_v: int, max_v: int) -> int:
    """Lee un entero desde input validando rango; repite hasta correcto."""
    while True:
        try:
            val = input(prompt).strip()
            iv = int(val)
            if iv < min_v or iv > max_v:
                print(f"Valor fuera de rango [{min_v}-{max_v}]. Intente de nuevo.")
                continue
            return iv
        except ValueError:
            print("Entrada inválida. Ingrese un número entero.")


def predict_manual(model_path: str = "model.pkl", scaler_path: str = "scaler.pkl") -> None:
    """Solicita valores, carga model y scaler, y muestra la probabilidad y resultado."""
    try:
        # Verificar existencia de archivos necesarios
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("No se encontró el modelo o el scaler. Entrene el modelo primero.")
            return

        # Leer inputs validados por rangos definidos en la especificación
        cpu = _read_int("CPU: ", 10, 100)
        ram = _read_int("RAM: ", 10, 100)
        temp = _read_int("Temperatura: ", 30, 100)
        net = _read_int("Red: ", 5, 100)

        # Cargar model y scaler
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Preparar vector de entrada y escalar
        X = np.array([[cpu, ram, temp, net]], dtype=float)
        X_scaled = scaler.transform(X)

        # Obtener probabilidad de clase positiva (fallo)
        prob = model.predict_proba(X_scaled)[0][1]
        pct = prob * 100.0

        # Determinar mensaje según umbral 0.5
        if prob >= 0.5:
            resultado = "RIESGO DE FALLO"
        else:
            resultado = "SERVIDOR ESTABLE"

        # Mostrar resultado con formato solicitado
        print(f"\nResultado: {resultado}\n")
        print(f"Probabilidad de fallo: {pct:.2f}%\n")

    except Exception as e:
        print(f"Error en la predicción: {e}")


if __name__ == "__main__":
    # Permite ejecutar predict.py de forma independiente
    predict_manual()
