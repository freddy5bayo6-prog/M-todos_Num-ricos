"""Punto de entrada del sistema de predicción de fallos.
Este módulo muestra el menú principal y delega en los módulos
responsables de cada funcionalidad.
Compatible con ejecución desde la terminal: python main.py
"""

# Importaciones necesarias
import sys
from generate_dataset import generate_dataset
from train import train_model
from utils import evaluate_model, show_statistics
from predict import predict_manual


def print_header() -> None:
    """Imprime el encabezado del sistema en consola."""
    # Encabezado visual del menú
    print("# =========================================")
    print("SISTEMA DE PREDICCIÓN DE FALLOS")
    print("MEDIANTE REGRESIÓN LOGÍSTICA\n")


def menu() -> None:
    """Bucle principal que muestra el menú y procesa opciones."""
    while True:
        try:
            print_header()
            # Opciones del menú
            print("1. Generar dataset")
            print("2. Entrenar modelo")
            print("3. Evaluar modelo")
            print("4. Realizar predicción")
            print("5. Mostrar estadísticas")
            print("6. Salir\n")

            opcion = input("Seleccione una opción: ").strip()

            if opcion == "1":
                # Generar dataset.csv con 500 registros
                generate_dataset(file_path="dataset.csv", n=500)

            elif opcion == "2":
                # Entrenar modelo y guardar model.pkl y scaler.pkl
                train_model(dataset_path="dataset.csv", model_path="model.pkl", scaler_path="scaler.pkl")

            elif opcion == "3":
                # Evaluar modelo en conjunto de prueba
                evaluate_model(dataset_path="dataset.csv", model_path="model.pkl", scaler_path="scaler.pkl")

            elif opcion == "4":
                # Realizar predicción manual
                predict_manual(model_path="model.pkl", scaler_path="scaler.pkl")

            elif opcion == "5":
                # Mostrar estadísticas del dataset
                show_statistics(dataset_path="dataset.csv")

            elif opcion == "6":
                # Salir con código 0
                print("Saliendo...")
                sys.exit(0)

            else:
                print("Opción no válida. Intente de nuevo.\n")

        except KeyboardInterrupt:
            # Manejo de Ctrl+C
            print("\nInterrupción por teclado. Saliendo...")
            sys.exit(0)
        except Exception as e:
            # Captura de errores inesperados para no romper la ejecución
            print(f"Ocurrió un error: {e}\n")


if __name__ == "__main__":
    # Punto de entrada cuando se ejecuta directamente
    menu()
