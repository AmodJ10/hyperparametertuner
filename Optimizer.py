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
            indis.append(indi)
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
            self.tournament()
        elif self.selection == 'roulette':
            self.roulette_wheel()
        else:
            if random.random() < self.selection_chance:
                self.tournament()
            else:
                self.roulette_wheel()

    def blend_crossover(parent1, parent2, alpha=0.5):
        child = {}
        for key in parent1:
            child[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
        return child

    def mutate(individual, mutation_rate=0.1):
        mutated_individual = {}
        for key, value in individual.items():
            if random.random() < mutation_rate:
                mutated_individual[key] = value + random.random(scale=0.1)
            else:
                mutated_individual[key] = value
        return mutated_individual
    
    def has_converged(best_fitness_values, tol=1e-5, patience=5):
        if len(best_fitness_values) < patience + 1:
            return False
        return best_fitness_values[-patience:] - best_fitness_values[-1:] < tol
    
