"""Módulo para generar `dataset.csv` con 500 registros.
La variable `Fallo` se calcula de forma determinista a partir
de las variables de entrada (CPU, RAM, Temperatura, Red).
"""

import os
import numpy as np
import pandas as pd


def generate_dataset(file_path: str = "dataset.csv", n: int = 500) -> None:
    """Genera un dataset y lo guarda en `file_path`.

    Parámetros:
    - file_path: ruta donde guardar el CSV.
    - n: número de registros a generar.
    """
    try:
        # Valores mínimos y máximos para cada característica
        cpu_min, cpu_max = 10, 100
        ram_min, ram_max = 10, 100
        temp_min, temp_max = 30, 100
        net_min, net_max = 5, 100

        # Semilla para reproducibilidad
        np.random.seed(42)

        # Generar valores enteros uniformes dentro de los rangos
        cpu = np.random.randint(cpu_min, cpu_max + 1, size=n)
        ram = np.random.randint(ram_min, ram_max + 1, size=n)
        temp = np.random.randint(temp_min, temp_max + 1, size=n)
        net = np.random.randint(net_min, net_max + 1, size=n)

        # Normalizar cada variable en rango [0,1] usando sus propios límites
        cpu_norm = (cpu - cpu_min) / (cpu_max - cpu_min)
        ram_norm = (ram - ram_min) / (ram_max - ram_min)
        temp_norm = (temp - temp_min) / (temp_max - temp_min)
        net_norm = (net - net_min) / (net_max - net_min)

        # Calcular una 'probabilidad' determinística promedio de fallo
        prob = (cpu_norm + ram_norm + temp_norm + net_norm) / 4.0

        # Etiquetado: si la probabilidad >= 0.6 -> fallo (1), sino 0
        fallo = (prob >= 0.6).astype(int)

        # Construir DataFrame con los nombres de columnas requeridos
        df = pd.DataFrame({
            "CPU": cpu,
            "RAM": ram,
            "Temperatura": temp,
            "Red": net,
            "Fallo": fallo,
        })

        # Guardar CSV en la ruta solicitada
        df.to_csv(file_path, index=False)

        # Mensajes solicitados por la especificación
        print("Dataset generado correctamente.")
        print(f"Total de registros: {len(df)}\n")
        # Mostrar primeras 10 filas
        print(df.head(10).to_string(index=False))

    except Exception as e:
        # Manejo de errores genérico para no detener el menú
        print(f"Error al generar el dataset: {e}")


if __name__ == "__main__":
    # Permite ejecutar directamente el script para pruebas rápidas
    generate_dataset()
