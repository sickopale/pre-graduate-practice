import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tabulate import tabulate

X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

POP_SIZE = 20
GENERATIONS = 50
MUTATION_RATE = 0.2
ELITISM = 2
SIZE_PENALTY_WEIGHT = 0.001
SIZE_MUTATION_RATE = 0.3
MIN_HIDDEN_LAYERS = 1
MAX_HIDDEN_LAYERS = 3
MIN_NEURONS_PER_LAYER = 2
MAX_NEURONS_PER_LAYER = 20


def create_mlp(hidden_layer_sizes=None):
    if hidden_layer_sizes is None:

        num_layers = random.randint(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS)
        hidden_layer_sizes = tuple(random.randint(MIN_NEURONS_PER_LAYER, MAX_NEURONS_PER_LAYER)
                                   for _ in range(num_layers))

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1, warm_start=True)
    mlp.fit(X_train[:5], y_train[:5])
    return mlp, hidden_layer_sizes


def flatten_weights(mlp):
    return np.hstack([w.flatten() for w in mlp.coefs_] + [b.flatten() for b in mlp.intercepts_])


def set_weights(mlp, flat_weights):
    shapes = [w.shape for w in mlp.coefs_] + [b.shape for b in mlp.intercepts_]
    split_points = np.cumsum([np.prod(shape) for shape in shapes])
    params = np.split(flat_weights, split_points[:-1])
    mlp.coefs_ = [param.reshape(shape) for param, shape in zip(params[:len(mlp.coefs_)], shapes[:len(mlp.coefs_)])]
    mlp.intercepts_ = [param.reshape(shape) for param, shape in zip(params[len(mlp.coefs_):], shapes[len(mlp.coefs_):])]
    return mlp


def count_parameters(mlp):
    return sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)


def weight_vector_size_kb(mlp):
    return flatten_weights(mlp).nbytes / 1024


def evaluate_fitness(mlp):
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    size_penalty = SIZE_PENALTY_WEIGHT * count_parameters(mlp)
    return accuracy - size_penalty


def select_best(population, scores, num_keep):
    sorted_indices = np.argsort(scores)[::-1]
    return [population[i] for i in sorted_indices[:num_keep]]


def crossover(parent1, parent2, parent1_struct, parent2_struct):
    min_len = min(len(parent1), len(parent2))
    child_weights = np.where(np.random.rand(min_len) < 0.5, parent1[:min_len], parent2[:min_len])

    if random.random() < 0.5:
        child_struct = parent1_struct
    else:
        child_struct = parent2_struct

    return child_weights, child_struct


def mutate_weights(weights, rate=0.2):
    mask = np.random.rand(len(weights)) < rate
    weights[mask] += np.random.normal(0, 0.1, np.sum(mask))
    return weights


def mutate_structure(hidden_layer_sizes):
    if random.random() < SIZE_MUTATION_RATE:
        mutation_type = random.choice(['add', 'remove', 'modify'])

        if mutation_type == 'add' and len(hidden_layer_sizes) < MAX_HIDDEN_LAYERS:
            new_layer_size = random.randint(MIN_NEURONS_PER_LAYER, MAX_NEURONS_PER_LAYER)
            insert_pos = random.randint(0, len(hidden_layer_sizes))
            hidden_layer_sizes = hidden_layer_sizes[:insert_pos] + (new_layer_size,) + hidden_layer_sizes[insert_pos:]
        elif mutation_type == 'remove' and len(hidden_layer_sizes) > MIN_HIDDEN_LAYERS:
            remove_pos = random.randint(0, len(hidden_layer_sizes) - 1)
            hidden_layer_sizes = hidden_layer_sizes[:remove_pos] + hidden_layer_sizes[remove_pos + 1:]
        elif mutation_type == 'modify':
            modify_pos = random.randint(0, len(hidden_layer_sizes) - 1)
            hidden_layer_sizes = list(hidden_layer_sizes)
            hidden_layer_sizes[modify_pos] = random.randint(MIN_NEURONS_PER_LAYER, MAX_NEURONS_PER_LAYER)
            hidden_layer_sizes = tuple(hidden_layer_sizes)

    return hidden_layer_sizes


def genetic_algorithm():
    population = []
    architectures = []
    for _ in range(POP_SIZE):
        mlp, arch = create_mlp()
        population.append(set_weights(mlp, np.random.randn(len(flatten_weights(mlp)))))
        architectures.append(arch)

    history = []

    for gen in range(GENERATIONS):
        scores = []
        accs = []
        for mlp in population:
            mlp.fit(X_train, y_train)
            scores.append(evaluate_fitness(mlp))
            accs.append(accuracy_score(y_test, mlp.predict(X_test)))

        best_score = max(accs)
        avg_score = np.mean(accs)
        worst_score = min(accs)

        history.append((gen + 1, best_score, avg_score, worst_score))
        print(
            f"Generation {gen + 1}/{GENERATIONS} | Best Acc: {best_score:.4f}, Avg: {avg_score:.4f}, Worst: {worst_score:.4f}")

        best_indices = np.argsort(scores)[::-1][:ELITISM]
        best_weights = [flatten_weights(population[i]) for i in best_indices]
        best_architectures = [architectures[i] for i in best_indices]

        next_generation = []
        next_architectures = []

        for i in best_indices:
            next_generation.append(population[i])
            next_architectures.append(architectures[i])

        while len(next_generation) < POP_SIZE:
            parent_indices = random.sample(range(len(best_weights)), 2)
            p1_weights, p2_weights = best_weights[parent_indices[0]], best_weights[parent_indices[1]]
            p1_arch, p2_arch = best_architectures[parent_indices[0]], best_architectures[parent_indices[1]]

            child_weights, child_arch = crossover(p1_weights, p2_weights, p1_arch, p2_arch)

            child_weights = mutate_weights(child_weights, MUTATION_RATE)
            child_arch = mutate_structure(child_arch)

            mlp, _ = create_mlp(child_arch)
            try:
                mlp = set_weights(mlp, child_weights[:len(flatten_weights(mlp))])
                next_generation.append(mlp)
                next_architectures.append(child_arch)
            except:
                pass

        population = next_generation
        architectures = next_architectures

    final_scores = []
    final_accs = []
    for mlp in population:
        mlp.fit(X_train, y_train)
        final_scores.append(evaluate_fitness(mlp))
        final_accs.append(accuracy_score(y_test, mlp.predict(X_test)))

    best_idx = np.argmax(final_scores)
    best_mlp = population[best_idx]
    final_accuracy = final_accs[best_idx]
    best_architecture = architectures[best_idx]

    return best_mlp, final_accuracy, history, best_architecture


print("\n===== Initial Model =====")
initial_mlp, _ = create_mlp((10, 5))
initial_mlp.fit(X_train, y_train)
initial_accuracy = accuracy_score(y_test, initial_mlp.predict(X_test))
initial_params = count_parameters(initial_mlp)
initial_size = weight_vector_size_kb(initial_mlp)

print(f"Accuracy: {initial_accuracy:.4f}")
print(f"Trainable Parameters: {initial_params}")
print(f"Weight Size: {initial_size:.2f} KB")
print(f"Architecture: {initial_mlp.hidden_layer_sizes}")

best_model, best_accuracy, history, best_arch = genetic_algorithm()
best_params = count_parameters(best_model)
best_size = weight_vector_size_kb(best_model)

gens, best_acc, avg_acc, worst_acc = zip(*history)
plt.figure(figsize=(10, 5))
plt.plot(gens, best_acc, label="Best Accuracy", marker="o")
plt.plot(gens, avg_acc, label="Average Accuracy", linestyle="--")
plt.plot(gens, worst_acc, label="Worst Accuracy", linestyle="dotted")
plt.xlabel("Generation")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy Evolution (with size mutation)")
plt.legend()
plt.grid()
plt.tight_layout()

summary_table = [
    ["Metric", "Before Optimization", "After Optimization"],
    ["Accuracy", f"{initial_accuracy:.4f}", f"{best_accuracy:.4f}"],
    ["Trainable Parameters", f"{initial_params}", f"{best_params}"],
    ["Weight Size (KB)", f"{initial_size:.2f}", f"{best_size:.2f}"],
    ["Architecture", initial_mlp.hidden_layer_sizes, best_arch],
]

print("\n===== Summary Table =====")
print(tabulate(summary_table, headers="first row", tablefmt="fancy_grid"))

plt.show()