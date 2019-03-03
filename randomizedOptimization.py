import mlrose
import numpy as np
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def neuralNetworkWeights():
    # This code was originally taken and modified from https://mlrose.readthedocs.io/en/stable/source/intro.html
    hidden_nodes = [2]

    NNRHC = mlrose.NeuralNetwork(hidden_nodes, activation='relu', algorithm='random_hill_climb', max_iters=100,
                                 bias=True, is_classifier=True, learning_rate=0.1, clip_max=10000000000.0,
                                 restarts=20, random_state=1, early_stopping=True)

    schedule = mlrose.ExpDecay(init_temp=10, exp_const=.1, min_temp=1)
    NNSA = mlrose.NeuralNetwork(hidden_nodes, activation='relu', algorithm='simulated_annealing', max_iters=100,
                                bias=True, is_classifier=True, learning_rate=0.1, clip_max=10000000000.0,
                                schedule=schedule, random_state=1, early_stopping=True)

    NNGA = mlrose.NeuralNetwork(hidden_nodes, activation='relu', algorithm='genetic_alg', max_iters=100,
                                bias=True, is_classifier=True, learning_rate=0.1, clip_max=10000000000.0,
                                pop_size=200, mutation_prob=0.9, random_state=1, early_stopping=True)

    mainData = load_breast_cancer()

    X = mainData['data']
    y = mainData['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    NNRHC.fit(X_train, y_train)
    NNSA.fit(X_train, y_train)
    NNGA.fit(X_train, y_train)

    predictionsRHC = NNRHC.predict(X_test)
    predictionsSA = NNSA.predict(X_test)
    predictionsGA = NNGA.predict(X_test)

    accuracyScoreRHC = accuracy_score(y_test, predictionsRHC) * 100
    accuracyScoreSA = accuracy_score(y_test, predictionsSA) * 100
    accuracyScoreGA = accuracy_score(y_test, predictionsGA) * 100

    print("Neural Network Accuracy: ")
    print('Random Hill Climbing: ')
    print("Accuracy: ", accuracyScoreRHC)

    print('Simulated Annealing: ')
    print("Accuracy: ", accuracyScoreSA)

    print('Genetic Algorithm: ')
    print("Accuracy: ", accuracyScoreGA)

    return accuracyScoreRHC, accuracyScoreSA, accuracyScoreGA


def radomHillClimb(fitness, x):
    # This code was originally taken and modified from https://mlrose.readthedocs.io/en/stable/source/intro.html
    start = time.time()

    # Initialize fitness function object using pre-defined class
    #fitness = mlrose.Queens()

    # Define optimization problem object
    if (x == 0):
        problem = mlrose.DiscreteOpt(length=12, fitness_fn=fitness, maximize=False, max_val=12)
    elif (x == 1):
        problem = mlrose.DiscreteOpt(length=9, fitness_fn=fitness, maximize=False, max_val=3)
    else:
        problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)

    # Solve using random hill climb
    if (x == 0):
        init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    elif (x == 1):
        init_state = np.array([0, 1, 2, 0, 1, 2, 0, 1, 1])
    else:
        init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, restarts=10, init_state=init_state,
                                                                       max_iters=1000, random_state=1, curve=True)

    end = time.time()

    print("Random Hill Climb:")
    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness)
    print("Time: " + str(end-start))
    return best_fitness, end-start


def simulatedAnnealing(fitness, x):
    # This code was originally taken and modified from https://mlrose.readthedocs.io/en/stable/source/intro.html
    start = time.time()

    # Initialize fitness function object using pre-defined class
    #fitness = mlrose.Queens()

    # Define optimization problem object
    if (x == 0):
        problem = mlrose.DiscreteOpt(length=12, fitness_fn=fitness, maximize=False, max_val=12)
    elif (x == 1):
        problem = mlrose.DiscreteOpt(length=9, fitness_fn=fitness, maximize=False, max_val=3)
    else:
        problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)

    # Define decay schedule
    schedule = mlrose.GeomDecay()

    # Solve using random hill climb
    if (x == 0):
        init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    elif (x == 1):
        init_state = np.array([0, 1, 2, 0, 1, 2, 0, 1, 1])
    else:
        init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=10,
                                                                         max_iters=1000, init_state=init_state,
                                                                         random_state=1, curve=True)

    end = time.time()

    print("Simulated Annealing:")
    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness)
    print("Time: " + str(end - start))
    return best_fitness, end - start


def geneticAlgorithm(fitness, x):
    # This code was originally taken and modified from https://mlrose.readthedocs.io/en/stable/source/intro.html
    start = time.time()

    # Initialize fitness function object using pre-defined class
    #fitness = mlrose.Queens()

    # Define optimization problem object
    if (x == 0):
        problem = mlrose.DiscreteOpt(length=12, fitness_fn=fitness, maximize=False, max_val=12)
    elif (x == 1):
        problem = mlrose.DiscreteOpt(length=9, fitness_fn=fitness, maximize=False, max_val=3)
    else:
        problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)

    # Solve using genetic algorithm
    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, curve=True,
                                                                 max_iters=1000, random_state=1)

    end = time.time()

    print("Genetic Algorithm:")
    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness)
    print("Time: " + str(end - start))
    return best_fitness, end - start


def MIMIC(fitness, x):
    # This code was originally taken and modified from https://mlrose.readthedocs.io/en/stable/source/intro.html
    start = time.time()

    # Initialize fitness function object using pre-defined class
    #fitness = mlrose.Queens()

    # Define optimization problem object
    if (x == 0):
        problem = mlrose.DiscreteOpt(length=12, fitness_fn=fitness, maximize=False, max_val=12)
    elif (x == 1):
        problem = mlrose.DiscreteOpt(length=9, fitness_fn=fitness, maximize=False, max_val=3)
    else:
        problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)


    # Solve using genetic algorithm
    best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.1,
                                                           curve=True, max_iters=10, random_state=1)

    end = time.time()

    print("MIMIC:")
    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness)
    print("Time: " + str(end - start))
    return best_fitness, end - start


# Calling main function
if __name__ == "__main__":
    aveAccuracyScoreRHC = 0
    aveAccuracyScoreSA = 0
    aveAccuracyScoreGA = 0
    for x in range(0, 10):
        accuracyScoreRHC, accuracyScoreSA, accuracyScoreGA = neuralNetworkWeights()
        aveAccuracyScoreRHC += accuracyScoreRHC
        aveAccuracyScoreSA += accuracyScoreSA
        aveAccuracyScoreGA += accuracyScoreGA

    aveAccuracyScoreRHC /= 10
    aveAccuracyScoreSA /= 10
    aveAccuracyScoreGA /= 10

    print('///////////////////////////////////////////////////////////////////////////////')
    print("OverAll Neural Network Accuracy: ")
    print('Random Hill Climbing: ')
    print("Accuracy: ", aveAccuracyScoreRHC)
    print('Simulated Annealing: ')
    print("Accuracy: ", aveAccuracyScoreSA)
    print('Genetic Algorithm: ')
    print("Accuracy: ", aveAccuracyScoreGA)
    print('///////////////////////////////////////////////////////////////////////////////')

    edges = [(0, 1), (1, 4), (1, 3), (2, 4), (3, 7), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (5, 3),
             (0, 3), (0, 2), (1, 7), (1, 6), (0, 4), (1, 2), (3, 4), (8, 0), (8, 4), (8, 2), (8, 1)]
    weights = [3, 4, 5, 7, 9, 6, 10, 11]
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    maxWeightPct = 2
    fitnessArray = [mlrose.Queens(), mlrose.MaxKColor(edges), mlrose.Knapsack(weights, values, maxWeightPct)]
    titleArray = ['Queens', 'Max Color', 'Knapsack']
    for x in range(len(fitnessArray)):
        aveFitnessRHC = 0
        aveFitnessSA = 0
        aveFitnessGA = 0
        aveFitnessM = 0
        aveTimeRHC = 0
        aveTimeSA = 0
        aveTimeGA = 0
        aveTimeM = 0
        for y in range(0, 10):
            print('Results for: ', titleArray[x])
            bestFitnessRHC, timeRHC = radomHillClimb(fitnessArray[x], x)
            bestFitnessSA, timeSA = simulatedAnnealing(fitnessArray[x], x)
            bestFitnessGA, timeGA = geneticAlgorithm(fitnessArray[x], x)
            bestFitnessM, timeM = MIMIC(fitnessArray[x], x)
            aveFitnessRHC += bestFitnessRHC
            aveFitnessSA += bestFitnessSA
            aveFitnessGA += bestFitnessGA
            aveFitnessM += bestFitnessM
            aveTimeRHC += timeRHC
            aveTimeSA += timeSA
            aveTimeGA += timeGA
            aveTimeM += timeM

        aveFitnessRHC /= 10
        aveFitnessSA /= 10
        aveFitnessGA /= 10
        aveFitnessM /= 10
        aveTimeRHC /= 10
        aveTimeSA /= 10
        aveTimeGA /= 10
        aveTimeM /= 10
        print('///////////////////////////////////////////////////////////////////////////////')
        print('The overall Results for ' + titleArray[x] + ' are: ')
        print('Average Fitness RHC: ', aveFitnessRHC)
        print('Average Time RHC: ', aveTimeRHC)
        print('Average Fitness SA: ', aveFitnessSA)
        print('Average Time SA: ', aveTimeSA)
        print('Average Fitness GA: ', aveFitnessGA)
        print('Average Time GA: ', aveTimeGA)
        print('Average Fitness MIMIC: ', aveFitnessM)
        print('Average Time MIMIC: ', aveTimeM)
        print('///////////////////////////////////////////////////////////////////////////////')

