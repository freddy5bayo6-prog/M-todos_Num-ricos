"""Script de prueba: ejecuta las acciones correspondientes a las opciones
1 (generar dataset), 2 (entrenar), 3 (evaluar) y 5 (estadísticas).
Usar para reproducir fallos sin interacción en `main.py`.
"""

from generate_dataset import generate_dataset
from train import train_model
from utils import evaluate_model, show_statistics


def run_all():
    print("== Ejecutando opción 1: generar dataset ==")
    generate_dataset(file_path="dataset.csv", n=500)

    print("\n== Ejecutando opción 2: entrenar modelo ==")
    train_model(dataset_path="dataset.csv", model_path="model.pkl", scaler_path="scaler.pkl")

    print("\n== Ejecutando opción 3: evaluar modelo ==")
    evaluate_model(dataset_path="dataset.csv", model_path="model.pkl", scaler_path="scaler.pkl")

    print("\n== Ejecutando opción 5: mostrar estadísticas ==")
    show_statistics(dataset_path="dataset.csv")


if __name__ == "__main__":
    run_all()
