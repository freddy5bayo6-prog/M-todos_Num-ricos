# Proyecto: Sistema de Predicción de Fallos mediante Regresión Logística

Este proyecto implementa una aplicación de consola (sin GUI) para generar
un dataset sintético, entrenar un modelo de Regresión Logística, evaluar
su rendimiento y realizar predicciones manuales desde la terminal de VS Code.

## Estructura del proyecto

proyecto-regresion-logistica/

- main.py
- generate_dataset.py
- train.py
- predict.py
- utils.py
- dataset.csv (generado por la opción 1)
- model.pkl (generado por la opción 2)
- scaler.pkl (generado por la opción 2)
- requirements.txt
- README.md

## Instalación de dependencias

Ejecute en la terminal integrada de VS Code (Windows):

```bash
pip install -r requirements.txt
```

## Cómo ejecutar

Abrir la terminal integrada en VS Code y ejecutar:

```bash
python main.py
```

Se mostrará un menú con las opciones:

1. Generar dataset
2. Entrenar modelo
3. Evaluar modelo
4. Realizar predicción
5. Mostrar estadísticas
6. Salir

### Ejemplo de flujo

1. Seleccione `1` para generar `dataset.csv` con 500 registros.
2. Seleccione `2` para entrenar el modelo (se generarán `model.pkl` y `scaler.pkl`).
3. Seleccione `3` para ver métricas de evaluación.
4. Seleccione `4` para ingresar manualmente CPU/RAM/Temperatura/Red y obtener
   la probabilidad de fallo.

## Descripción del dataset

- Columnas: `CPU`, `RAM`, `Temperatura`, `Red`, `Fallo`.
- Rangos:
  - `CPU`: 10 a 100
  - `RAM`: 10 a 100
  - `Temperatura`: 30 a 100
  - `Red`: 5 a 100
- `Fallo` es una etiqueta determinística derivada de las otras columnas:
  se normalizan las variables al rango [0,1] y se calcula la media; si la
  media >= 0.6, se marca como `Fallo=1`, en caso contrario `0`.

## Métricas mostradas

- Accuracy: proporción de predicciones correctas.
- Precision: proporción de verdaderos positivos sobre positivos predichos.
- Recall: proporción de verdaderos positivos sobre positivos reales.
- F1 Score: media armónica entre precision y recall.
- Matriz de Confusión: tabla con verdaderos/falsos positivos/negativos.

## Ejemplos de uso

- Generar dataset:

```bash
python main.py
# luego seleccionar 1
```

- Entrenar modelo:

```bash
python main.py
# luego seleccionar 2
```

- Predecir manual:

```bash
python main.py
# luego seleccionar 4 e ingresar valores solicitar
```

## Notas finales

- El proyecto está pensado para demostraciones en consola y es compatible
  con Python 3.12.
- El código incluye manejo básico de excepciones y mensajes claros para
  guiar al usuario en caso de pasos previos faltantes (dataset o modelo).
