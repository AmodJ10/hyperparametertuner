from sklearn.model_selection import KFold, cross_val_score
from individual import Individual
from sklearn.base import clone
import random
import bisect

class Optimizer:
    def __init__(self,X,y,model,ranges,selection="both",population=10,n_splits = 5,selection_chance=0.5,tournament_split=2):
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.X = X
        self.y = y
        self.model = model
        self.ranges = ranges
        self.selection = selection
        self.tournament_split = tournament_split
        self.population = population
        self.selection_chance = selection_chance
        self.indis = []
        self.roulette_probs = []
    
    def getScore(self,indi):
        new_model = clone(self.model)
        new_params = new_model.get_params()
        
        for param, value in indi.params.items():
            new_params[param] = value
        
        new_model.set_params(**new_params)
        cv_scores = cross_val_score(new_model, self.X, self.y, cv=self.kfold, scoring='accuracy')
        return cv_scores.mean()

    def spawn(self):
        indis = []
        ranges = self.ranges
        for _ in range(self.population):
            indi = {}
            for key,value in ranges.items():
                if value[1] is int:
                    indi[key] = [random.randint(value[0][0],value[0][1]),value[1]]
                else:
                    indi[key] = [random.uniform(value[0][0],value[0][1]),value[1]]
            indis.append(Individual(indi))      # Added initialisation of Individual
        self.indis = indis
        self.sort_indis()

    def sort_indis(self):
        indis = self.indis
        for i in indis:
            if i.score == -1:
                i.score = self.getScore(i)
        indis.sort(key=lambda x: x.score)
        self.indis = indis
        sum_score = sum((i.score for i in indis))
        roulette_probs = [i.score/sum_score for i in indis]
        roulette_prob_sum = 0
        for i in range(len(roulette_prob_sum)):
            roulette_prob_sum += roulette_probs[i]
            roulette_probs[i] = roulette_prob_sum
        self.roulette_probs = roulette_probs
    
    def tournament(self):
        tournament_indis = random.sample(range(len(self.indis)),self.tournament_split)
        tournament_indis.sort()
        selected_indi_1 = self.indis[tournament_indis[-1]]
        selected_indi_2 = self.indis[tournament_indis[-2]]
        return selected_indi_1,selected_indi_2
    
    def roulette_wheel(self):
        selected_indi_1 = self.indis[bisect.bisect_left(self.roulette_probs,random.random())]
        selected_indi_2 = self.indis[bisect.bisect_left(self.roulette_probs,random.random())]
        return selected_indi_1,selected_indi_2
    
    def select(self):
        if self.selection == 'tournament':
            return self.tournament()    # Added return statement
        elif self.selection == 'roulette':
            return self.roulette_wheel() # Added return statement
        else:
            if random.random() < self.selection_chance:
                return self.tournament() # Added return statement
            else:
                return self.roulette_wheel()  # Added return statement

    def uniform_crossover(parent1, parent2):
    child_gene =  {}
    for key in parent1.gene:
        if np.random.rand() < 0.5:
            child_gene[key] = parent1.gene[key]
        else:
            child_gene[key] = parent2.gene[key]
    return Individual(child_gene)

def blend_crossover(parent1, parent2, alpha=0.5):
    child_gene = {}
    for key in parent1.gene:
        value_type = parent1.gene[key][1]
        if value_type == bool:
            child_gene[key] = random.choice([parent1.gene[key], parent2.gene[key]])
        else:
            rand_val = random.uniform(-alpha * diff, (1 + alpha) * diff)
            value = parent1.gene[key][0] + rand_val

            child_gene[key] = [value, value_type]

    return Individual(child_gene)

def crossover(parent1, parent2, crossover_prob=0.5):
    if np.random.rand() < crossover_prob:
        return uniform_crossover(parent1, parent2)
    else:
        return blend_crossover(parent1, parent2)

    def mutation(individual, mutation_prob=0.1):
    mutated_gene = {}
    for key in individual.gene:
        if np.random.rand() < mutation_prob:
            value_type = individual.ranges[key][1]
            low, high = individual.ranges[key][0]
            
            if value_type == bool:
                mutated_gene[key] = not individual.gene[key]
            elif value_type == int:
                mutated_gene[key] = random.randint(low,high)
            else:
                mutated_gene[key] = random.uniform(low, high)
        else:
            mutated_gene[key] = individual.gene[key]

    return Individual(mutated_gene)
    
    def has_converged(best_fitness_values, tol=1e-5, patience=5):
        try:
            if len(best_fitness_values) < patience + 1:
                return False
            return ((best_fitness_values[-patience:] - best_fitness_values[-1:]) < tol)
        except TypeError:
            # Handle the case where best_fitness_values is not a numerical array
            return False
        except Exception as e:
            # Handle other unexpected exceptions
            print(f"An error occurred: {e}")
            return False
    
