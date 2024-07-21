import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt

# Emergency scores matrix
emergency_scores = np.array([
    [5, 2, 4, 8, 9, 0, 3, 3, 8, 7],
    [5, 5, 3, 4, 4, 6, 4, 1, 9, 1],
    [4, 1, 2, 1, 3, 8, 7, 8, 9, 1],
    [1, 7, 1, 6, 9, 3, 1, 9, 6, 9],
    [4, 7, 4, 9, 9, 8, 6, 5, 4, 2],
    [7, 5, 8, 2, 5, 2, 3, 9, 8, 2],
    [1, 4, 0, 6, 8, 4, 0, 1, 2, 1],
    [1, 5, 2, 1, 2, 8, 3, 3, 6, 2],
    [4, 5, 9, 6, 3, 9, 7, 6, 5, 10],
    [0, 6, 2, 8, 7, 1, 2, 1, 5, 3]
])

num_generations = 100
population_size = 100
mutation_rate = 0.1
crossover_rate = 0.7

# Generate random initial population
def generate_initial_population(size):
    return [(random.randint(0, 9), random.randint(0, 9)) for _ in range(size)]

# Calculate cost
def calculate_cost(coordinate):
    cost = 0
    for i in range(emergency_scores.shape[0]):
        for j in range(emergency_scores.shape[1]):
            distance = sqrt((coordinate[0] - i)**2 + (coordinate[1] - j)**2)
            cost += distance * emergency_scores[i][j]
    return cost


# Uniform Crossover (CX)
def crossover(parent1, parent2):
    if random.random() > crossover_rate:
        return parent1 if random.random() < 0.5 else parent2  

    child = [None, None]

    for i in range(2):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]

    return tuple(child)

# Interchanging mutation
def mutate(coordinate):
    if random.random() < mutation_rate:
        # Swap x and y coordinates for interchanging mutation
        return (coordinate[1], coordinate[0])
    return coordinate

fitness_cache = {}

def precalculate_fitness():
    for i in range(10):  
        for j in range(10):
            fitness_cache[(i, j)] = calculate_cost((i, j))

def genetic_algorithm():
    costs = []
    generations_without_improvement = 0  
    stagnation_limit = 5  # Define the number of consecutive generations without improvement to consider stopping

    # Initialize the first generation
    parent1 = (random.randint(0, 9), random.randint(0, 9))
    parent2 = (random.randint(0, 9), random.randint(0, 9))

    offspring = crossover(parent1, parent2)
    offspring = mutate(offspring)
    offspring_cost = fitness_cache.get(offspring, calculate_cost(offspring))

    print(f'Generation\tProposed Coordinate\tCost Value\n1\t\t({offspring[0]+1},{offspring[1]+1})\t\t\t{offspring_cost:.5f}')

    costs.append(offspring_cost)


    for generation in range(2, num_generations + 1):
        # Get new parent for GA
        new_parent = None
        while not new_parent:
            random_coord = (random.randint(0, 9), random.randint(0, 9))
            random_coord_cost = fitness_cache.get(random_coord, calculate_cost(random_coord))
            if random_coord_cost is not None and random_coord_cost <= offspring_cost:
                new_parent = random_coord
 

        offspring = crossover(offspring, new_parent)
        offspring = mutate(offspring)
        offspring_cost = fitness_cache.get(offspring, calculate_cost(offspring))

        # Update current best solution
        if offspring_cost < min(costs):
            best_offspring = offspring
            best_cost = offspring_cost
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        print(f'{generation}\t\t({offspring[0]+1},{offspring[1]+1})\t\t\t{offspring_cost:.5f}')
        costs.append(offspring_cost)

        if generations_without_improvement >= stagnation_limit:
            break  
 

    return costs


precalculate_fitness()

costs = genetic_algorithm()

best_coordinate = None
best_cost = float('inf')
for i in range(10):
    for j in range(10):
        cost = fitness_cache[(i, j)]
        if cost < best_cost:
            best_cost = cost
            best_coordinate = (i, j)

print(f'\nBest Cost: {best_cost:.5f}')
print(f'Coordinate: ({best_coordinate[0]+1},{best_coordinate[1]+1})')
best_emergency_score = emergency_scores[best_coordinate]
print(f'Emergency Score at Best Coordinate: {best_emergency_score}')

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))

# Plot costs over generations
ax1.plot(costs, 'o', markersize=4, color='blue')
ax1.plot(costs, color='blue', linewidth=2.5)
ax1.set_xlabel('Generation')
ax1.set_ylabel('Cost')
ax1.set_title('Cost over Generations')
ax1.grid(True)

# Plot the grid without squares
ax2.imshow(np.zeros((10, 10)), cmap='gray', alpha=0)  # Use cmap='gray' for black and white
ax2.invert_yaxis()  # Invert y-axis to show 1-10 going down
ax2.set_xticks(np.arange(10) + 0.5)
ax2.set_yticks(np.arange(10) + 0.5)
ax2.set_xticklabels(range(1, 11))
ax2.set_yticklabels(range(10, 0, -1))  # Reverse the y-axis labels
ax2.grid(color='black', linestyle='-', linewidth=2)
ax2.tick_params(axis='both', which='both', length=0)
ax2.set_aspect('equal')

# Insert numbers in the grid
for i in range(10):
    for j in range(10):
        score = emergency_scores[9 - i, j]  # Reverse the row index to match the grid
        color = 'black' 
        ax2.text(j, i, str(score), ha='center', va='center', color=color)

# Mark proposed coordinate with a marker
ax2.plot(best_coordinate[1], 9 - best_coordinate[0], marker='o', markersize=10, color='yellow', markeredgecolor='black', linestyle='None')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

