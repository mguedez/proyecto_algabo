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
    """Genera un individuo completamente aleatorio."""
    return {k: random.choice(v) for k, v in param_space.items()}


# Diversidad de la población

def population_diversity(pop):
    """
    Mide la diversidad como el porcentaje de genes distintos dentro de la población.
    Cuanto más cerca de 1, más diversa es la población.
    """
    diffs = []
    keys = list(param_space.keys())

    for i in range(len(pop)):
        for j in range(i+1, len(pop)):
            diff_count = sum(1 for k in keys if pop[i][k] != pop[j][k])
            diffs.append(diff_count / len(keys))

    return np.mean(diffs) if diffs else 0


# Cruce

def crossover(p1, p2):
    """
    Cruce equilibrado:
    - 80% hereda de los padres
    - 20% es reemplazado por un valor aleatorio (exploración)
    """
    child = {}

    for k, choices in param_space.items():
        r = random.random()
        if r < 0.4:
            child[k] = p1[k]
        elif r < 0.8:
            child[k] = p2[k]
        else:
            child[k] = random.choice(choices)

    return child


# Mutación adaptativa

def mutate(params, mutation_rate):
    """
    Mutación en varios parámetros.
    """
    new_params = params.copy()
    for k, choices in param_space.items():
        if random.random() < mutation_rate:
            new_params[k] = random.choice(choices)
    return new_params


# Algoritmo genético

def genetic_optimize(
    generations=25,
    population_size=30,
    elitism=0.1,
    base_mutation=0.25,
    immigrants_rate=0.1,
):
    
    population = [random_params() for _ in range(population_size)]
    best = None
    no_improve = 0  # para detectar mesetas

    for gen in range(generations):

        # Evaluación
        scores = [(evaluate_model(p), p) for p in population]
        scores.sort(reverse=True, key=lambda x: x[0])
        
        best_score, best_params = scores[0]

        # Detectar si quedó estancado
        if best is None or best_score > best[0]:
            best = (best_score, best_params)
            no_improve = 0
        else:
            no_improve += 1
        
        # Mutación adaptativa: si está estancado, aumentar mutación
        mutation_rate = base_mutation + min(0.4, no_improve * 0.05)

        diversity = population_diversity([p for _, p in scores])

        print(f"""
            === GENERACIÓN {gen+1} ===
            Mejor R²: {best_score:.4f}
            Diversidad: {diversity:.3f}
            Mutación usada: {mutation_rate:.2f}
            Mejores params: {best_params}
            """)

        # ---- Selección y elitismo ----
        n_elite = int(population_size * elitism)
        next_gen = [p for _, p in scores[:n_elite]]

        # ---- Creación de descendencia ----
        while len(next_gen) < population_size:
            p1, p2 = random.sample(scores[:15], 2)  # torneo entre los 15 mejores
            child = mutate(crossover(p1[1], p2[1]), mutation_rate)
            next_gen.append(child)

        # ---- Random immigrants ----
        num_imm = int(population_size * immigrants_rate)
        immigrants = [random_params() for _ in range(num_imm)]

        # Reemplazar a los peores
        next_gen[-num_imm:] = immigrants

        population = next_gen

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

print("\n=== OPTIMIZACIÓN GENÉTICA ===")
best_ga_score, best_ga_params = genetic_optimize(generations=10, population_size=40, elitism=0.05)

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
