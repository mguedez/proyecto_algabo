# ================================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS CON GA + B&B
# Dataset: California Housing
# Autor: Manuela Guedez Leivas y Lucía Olivera Freire
# ================================================

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


# ================================================
# 1. CARGA Y PREPROCESAMIENTO
# ================================================

housing = pd.read_csv("housing.csv").dropna()
RANDOM_STATE = 127

X = housing.drop(columns=["median_house_value"])
y = housing["median_house_value"]

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Batch del 10% para evaluación en GA
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
# 2. FUNCIÓN DE EVALUACIÓN (RMSE NEGATIVO)
# ================================================

def fitness_rmse(params, Xb, yb):
    model = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(**params, random_state=RANDOM_STATE))
    ])
    model.fit(Xb, yb)
    pred = model.predict(Xb)
    rmse = root_mean_squared_error(yb, pred) # comparación entre el resultado esperado y el obtenido de la predicción
    return -rmse  # para maximizar el fitness


# ================================================
# 3. BÚSQUEDA
# ================================================

param_space = {
    "n_estimators": [50, 100, 150, 200, 250, 300],
    "max_depth": [4, 6, 8, 10, 12, 14, None],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [1, 2, 3, 4, 5]
}


# ================================================
# 4. OPERADORES GENÉTICOS
# ================================================

def random_params():
    """
    Genera un conjunto aleatorio de hiperparámetros.

    Es un individuo para el algoritmo genético.
    """
    return {k: random.choice(v) for k, v in param_space.items()}

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


# ================================================
# 5. GA con patience
# ================================================

def genetic_optimize(
    generations=100, 
    population_size=40, 
    elitism=0.1, 
    patience=10,
    min_improvement=0.01   # mejora mínima del X%
):
    """
    min_improvement = 0.01 → 1% de mejora mínima requerida
    patience = 10 → cortar si no mejora en 10 generaciones
    """

    population = [random_params() for _ in range(population_size)]
    
    best_score = -np.inf
    best_params = None

    # Para patience
    generations_without_improvement = 0

    for gen in range(generations):

        # Evaluar población usando batch + RMSE
        # medimos el fitness de cada individuo
        scores = [(fitness_rmse(p, X_batch, y_batch), p) for p in population] 
        # Ordenar por fitness
        scores.sort(reverse=True, key=lambda x: x[0])

        gen_best_score, gen_best_params = scores[0]

        print(f"[GA] Generación {gen+1} | Mejor -RMSE: {gen_best_score:.4f} | {gen_best_params}")

        # -------- PATIENCE CHECK --------
        if best_score == -np.inf:
            # Primera iteración
            best_score = gen_best_score
            best_params = gen_best_params
        else:
            # ¿Hubo mejora superior al X%?
            improvement = (gen_best_score - best_score) / abs(best_score)

            if improvement >= min_improvement:
                best_score = gen_best_score
                best_params = gen_best_params
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # ¿Cortar?
            if generations_without_improvement >= patience:
                print(f"\n⛔ Early Stopping: no hubo mejora > {min_improvement*100:.1f}% "
                      f"en {patience} generaciones.")
                return best_score, best_params
        # --------------------------------

        # Elitismo
        n_elite = int(population_size * elitism)
        next_gen = [p for _, p in scores[:n_elite]]

        # Reproducción
        while len(next_gen) < population_size:
            p1, p2 = random.sample(population, 2)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)

        population = next_gen

    return best_score, best_params


# ================================================
# 6. BRANCH & BOUND (usa RMSE ahora)
# ================================================

def branch_and_bound(param_space, partial_params=None, best_score=-np.inf, memo=None):
    if partial_params is None:
        partial_params = {}
    if memo is None:
        memo = {}

    # parte que emula programación dinámica
    if len(partial_params) == len(param_space): # si ya tengo todos los parámetros (último nivel de la recursión)
        key = tuple(sorted(partial_params.items())) # armo la key para el memo
        if key in memo: # si ya existe, no recalculo de nuevo
            return memo[key]
        # si no existe, calculo el score
        score = fitness_rmse(partial_params, X_batch, y_batch)
        memo[key] = (partial_params, score) # lo guardo en el memo
        return partial_params, score

    remaining_keys = [k for k in param_space.keys() if k not in partial_params] # los hiperparámetros que faltan
    next_key = remaining_keys[0]

    best_local = (None, best_score)

    for val in param_space[next_key]: # para cada valor posible del hiperparámetro
        candidate = partial_params.copy() # {"arboles": 2, "profundidad": 4}
        candidate[next_key] = val # {"arboles": 2, "profundidad": 4, "min_samples_split": 6} -> min_samples_split es el next_key

        # es el partial_params de la llamada anterior cuando se completa el árbol y entra a la condición de PD
        # solo obtiene resultado != None cuando llega al ultimo nivel
        result, score = branch_and_bound(param_space, candidate, best_score, memo)

        if score > best_local[1]:
            best_local = (result, score)
            best_score = score

    return best_local

"""
¿POR QUÉ NO USAMOS PROGRAMACIÓN DINAMICA?

Pero el tuning de hiperparámetros NO tiene estructura de DP:
- no existe una recurrencia que relacione “el mejor bosquete parcial de hiperparámetros”
- no podés estimar el RMSE de una configuración parcial sin entrenar el modelo completo
- no hay forma de “combinar soluciones parciales”
- cada evaluación depende de todos los hiperparámetros a la vez
"""


# ================================================
# 7. EJECUCIÓN
# ================================================

print("\n=== OPTIMIZACIÓN GENÉTICA ===")

best_ga_score, best_ga_params = genetic_optimize(
    generations=100,
    population_size=40,
    elitism=0.1,
    patience=50,
    min_improvement=0.01   # 1%
)

# print("\n=== OPTIMIZACIÓN BRANCH AND BOUND ===")
# best_bb_params, best_bb_score = branch_and_bound(param_space)


# ================================================
# 8. RESULTADOS
# ================================================

print("\nRESULTADOS FINALES")
print("------------------")
print(f"Mejor fitness (GA): {-best_ga_score:.4f} RMSE")
print(f"Hiperparámetros (GA): {best_ga_params}")

# print(f"\nMejor fitness (B&B): {-best_bb_score:.4f} RMSE")
# print(f"Hiperparámetros (B&B): {best_bb_params}")

final_model = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(**best_ga_params, random_state=RANDOM_STATE))
])

final_model.fit(X_train, y_train)
test_score = final_model.score(X_test, y_test)
print(f"\nR² final en test (GA): {test_score:.4f}")
