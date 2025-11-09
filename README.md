# Optimización de Hiperparámetros de Random Forest (California Housing)

Proyecto académico de la asignatura "Algoritmos Avanzados de Búsqueda y Optimización" cuyo foco es comparar dos estrategias de búsqueda para ajustar hiperparámetros de un modelo de regresión `RandomForestRegressor`: un Algoritmo Genético (GA) y un esquema de exploración sistemática tipo Branch & Bound (B&B). El objetivo práctico es observar diferencias en exploración del espacio, convergencia y calidad final (R²) usando un dataset con tamaño suficiente para que los cambios sean significativos.

## Objetivo

Maximizar el coeficiente de determinación (R²) ajustando parámetros estructurales del Random Forest: número de árboles, profundidad máxima y criterios de partición/hojas.

## Estructura del repositorio

- `pipeline.py`: orquestación completa (preprocesamiento, GA, B&B, modelo final).
- `housing.csv`: dataset principal.
- `requirements.txt`: dependencias mínimas.

## Instalación (Windows / cmd)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bat
python pipeline.py
```

Se mostrará el progreso del GA por generaciones, luego la búsqueda B&B y finalmente un resumen de resultados + score en test usando la mejor configuración GA.
