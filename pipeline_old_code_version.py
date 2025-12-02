# ================================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS CON ALGORITMO GENÉTICO Y BRANCH & BOUND
# Dataset: California Housing
# Autor: Manuela Guedez Leivas y Lucía Olivera Freire
# ================================================

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
# ================================================

# Cargar dataset
housing = pd.read_csv("housing.csv").dropna()

RANDOM_STATE = 127

# Variables predictoras y target
X = housing.drop(columns=["median_house_value"])
y = housing["median_house_value"]

# División en train/test (sin estratificación porque es regresión)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# definimos batch para validación durante la optimización
X_train, X_batch, y_train, y_batch = train_test_split(
    X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
)

# Preprocesamiento
num_features = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income"
]
cat_features = ["ocean_proximity"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# ================================================
# 2. FUNCIÓN DE EVALUACIÓN COMÚN
# ================================================

def evaluate_model(params):
    """Evalúa un modelo de RandomForest con validación cruzada (regresión)"""
    model = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(**params, random_state=RANDOM_STATE))
    ])
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    score = np.mean(cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, scoring='r2'))
    return score

def fitness_rmse(params, X_batch, y_batch):
    """
    Función fitness que evalúa un individuo (configuración de hiperparámetros)
    usando RMSE en un batch de datos.
    
    Args:
        params (dict): Diccionario con hiperparámetros del RandomForest
        X_batch (array-like): Características del batch de entrenamiento
        y_batch (array-like): Target del batch de entrenamiento
    
    Returns:
        float: Negativo del RMSE (para maximizar fitness, minimizamos error)
               Valores más cercanos a 0 indican mejor ajuste.
    """
    model = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(**params, random_state=RANDOM_STATE))
    ])
    
    # Entrenar en el batch
    model.fit(X_batch, y_batch)
    
    # Predecir en el mismo batch
    y_pred = model.predict(X_batch)
    
    # Calcular RMSE
    rmse = np.sqrt(np.mean((y_batch - y_pred) ** 2))
    
    # Retornar negativo para convertir minimización en maximización
    # (los algoritmos genéticos típicamente maximizan fitness)
    return -rmse

# Espacio de búsqueda
param_space = {
    "n_estimators": [50, 100, 150, 200, 250, 300],
    "max_depth": [4, 6, 8, 10, 12, 14, None],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [1, 2, 3, 4, 5]
}

# ================================================
# 3. ALGORITMO GENÉTICO
# ================================================

def random_params():
    """
    Genera un conjunto aleatorio de hiperparámetros.

    Es un individuo para el algoritmo genético.
    """
    params = {k: random.choice(v) for k, v in param_space.items()}
    return params

def mutate(params, mutation_rate=0.4):
    """
    Mutación aleatoria de un parámetro.
    
    Params es el individuo a mutar.
    """
    new_params = params.copy()
    if random.random() < mutation_rate:
        key = random.choice(list(param_space.keys()))
        new_params[key] = random.choice(param_space[key])
    return new_params

def crossover(p1, p2):
    """Cruza dos padres para generar un hijo"""
    return {k: random.choice([p1[k], p2[k]]) for k in param_space.keys()}

def genetic_optimize(generations=8, population_size=10, elitism=0.4):
    """
    Algoritmo genético :)
    """
    population = [random_params() for _ in range(population_size)]
    best = None

    for gen in range(generations):
        scores = [(evaluate_model(p), p) for p in population]
        scores.sort(reverse=True, key=lambda x: x[0])

        best_score, best_params = scores[0]
        print(f"[GA] Generación {gen+1} | Mejor Score (R²): {best_score:.4f} | {best_params}")

        # Elitismo: conservar los mejores
        n_elite = int(population_size * elitism)
        next_gen = [p for _, p in scores[:n_elite]]

        # Rellenar con descendientes
        while len(next_gen) < population_size:
            p1, p2 = random.sample(next_gen, 2)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)

        population = next_gen
        best = (best_score, best_params)

    return best

# ================================================
# 4. BRANCH AND BOUND
# ================================================

def branch_and_bound(param_space, partial_params=None, best_score=-np.inf, memo=None):
    """Optimización de hiperparámetros mediante Branch and Bound"""
    if partial_params is None:
        partial_params = {}
    if memo is None:
        memo = {}

    # Caso base: configuración completa
    if len(partial_params) == len(param_space):
        key = tuple(sorted(partial_params.items()))
        if key in memo:
            return memo[key]
        score = evaluate_model(partial_params)
        memo[key] = (partial_params, score)
        return partial_params, score

    remaining_keys = [k for k in param_space.keys() if k not in partial_params]
    next_key = remaining_keys[0]

    best_local = (None, best_score)

    for val in param_space[next_key]:
        candidate = partial_params.copy()
        candidate[next_key] = val

        result, score = branch_and_bound(param_space, candidate, best_score, memo)

        if score > best_local[1]:
            best_local = (result, score)
            best_score = score

    return best_local

# ================================================
# 5. EJECUCIÓN Y COMPARACIÓN DE MÉTODOS
# ================================================

import time

start = time.perf_counter()
print("\n=== OPTIMIZACIÓN GENÉTICA ===")
best_ga_score, best_ga_params = genetic_optimize(generations=10, population_size=40, elitism=0.05)
end = time.perf_counter()

print(f"Tiempo de ejecució AG: {end - start:.4f} segundos")

print("\n=== OPTIMIZACIÓN BRANCH AND BOUND ===")
best_bb_params, best_bb_score = branch_and_bound(param_space)

# ================================================
# 6. RESULTADOS
# ================================================

print("\nRESULTADOS FINALES")
print("------------------")
print(f"Mejor (Algoritmo Genético): {best_ga_score:.4f}")
print(f"Hiperparámetros: {best_ga_params}")

print(f"\nMejor (Branch and Bound): {best_bb_score:.4f}")
print(f"Hiperparámetros: {best_bb_params}")

# ================================================
# 7. ENTRENAR MODELO FINAL CON MEJORES PARÁMETROS
# ================================================

final_model = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(**best_ga_params, random_state=RANDOM_STATE))
])
final_model.fit(X_train, y_train)
test_score = final_model.score(X_test, y_test)
print(f"\nR² final en test (usando GA): {test_score:.4f}")
