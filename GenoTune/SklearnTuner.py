from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
import random
import bisect
from joblib import Parallel, delayed

class Individual:
    def __init__(self,gene):
        self.gene = gene
        self.score = -1

class SklearnTuner:
    def __init__(
            self,
            X,
            y,
            model,
            model_type,
            param_distributions,
            parllel_computing = True,
            max_generation=10,
            selection="both",
            population=10,
            n_splits = 5,
            selection_chance=0.5,
            mutation_chance=0.1,
            new_population_chance=0.1,
            tournament_split=2,
            convergence_tolerance = 1e-5
            ):
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.X = X
        self.y = y
        self.model = model
        self.model_type = model_type
        self.max_generation = max_generation
        self.param_distributions = param_distributions
        self.parallel_computing = parllel_computing
        self.selection = selection
        self.tournament_split = tournament_split
        self.population = population
        self.selection_chance = selection_chance
        self.mutation_chance = mutation_chance
        self.new_population_chance = new_population_chance
        self.convergence_tolerance = convergence_tolerance
        self.indis = []
        self.roulette_probs = []
        self.best_scores = []
        self.average_scores = []
        self.new_population_ratio = 1/5
 
    def getScore(self,indi):
        if indi.score != -1:
            return indi.score
        new_model = clone(self.model)
        new_params = new_model.get_params()

        for param, value in indi.gene.items():
            new_params[param] = value    

        new_model.set_params(**new_params)
        if self.model_type == "regression":
            cv_scores = cross_val_score(new_model, self.X, self.y, cv=self.kfold, scoring='neg_mean_squared_error')
            indi.score = cv_scores.mean()
            return indi.score
        else:
            cv_scores = cross_val_score(new_model, self.X, self.y, cv=self.kfold, scoring='accuracy')
            indi.score = cv_scores.mean()
            return indi.score

    def get_new_indi(self,param_distributions):
        indi = {}
        for key,value in param_distributions.items():
            value_type = param_distributions[key][1]
            low, high = param_distributions[key][0][0], param_distributions[key][0][1]
            if value_type == bool:
                indi[key] = random.choice((True,False))
            elif value_type == str:
                indi[key] = random.choice(param_distributions[key][0])
            elif value[1] == int:
                indi[key] = random.randint(low,high)
            else:
                indi[key] = random.uniform(low,high)
        return indi

    def spawn(self):
        indis = []
        param_distributions = self.param_distributions
        for _ in range(self.population):
            indi = self.get_new_indi(param_distributions)
            indis.append(Individual(indi))
        self.indis = indis
        self.sort_indis()

    def sort_indis(self):
        indis = self.indis
        if self.parallel_computing:
           scores = Parallel(n_jobs=-1)(delayed(self.getScore)(i) for i in indis)
           for i, score in enumerate(scores):
               indis[i].score = score
        else:
            for i in indis:
                i.score = self.getScore(i)
            
        indis.sort(key=lambda x: x.score)
        indis = indis[-self.population:]

        if self.selection != 'tournament':
            self.precomputeRouletteWheelProbs(indis)
        
        self.indis = indis
    
    def precomputeRouletteWheelProbs(self,indis):
        sum_score = sum((i.score for i in indis))
        self.best_scores.append(indis[-1].score)
        self.average_scores.append(sum_score/self.population)

        roulette_probs = [i.score/sum_score for i in indis]
        roulette_prob_sum = 0
        
        for i in range(len(roulette_probs)):
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
            return self.tournament()
        elif self.selection == 'roulette':
            return self.roulette_wheel()
        else:
            if random.random() < self.selection_chance:
                return self.tournament()
            else:
                return self.roulette_wheel()

    def uniform_crossover(self,parent1, parent2):
        child_gene =  {}
        for key in parent1.gene:
            if random.random() < 0.5:
                child_gene[key] = parent1.gene[key]
            else:
                child_gene[key] = parent2.gene[key]
        return Individual(child_gene)

    def blend_crossover(self,parent1, parent2, alpha=0.5):
        child_gene = {}
        for key in parent1.gene:
            value_type = self.param_distributions[key][1]
            if value_type == bool:
                child_gene[key] = random.choice([parent1.gene[key], parent2.gene[key]])
            elif value_type == str:
                child_gene[key] = random.choice([parent1.gene[key], parent2.gene[key]])
            elif value_type == int:
                value = int(alpha * parent1.gene[key] + (1 -alpha) * parent2.gene[key])
                child_gene[key] = value
            else:
                value = alpha * parent1.gene[key] + (1 -alpha) * parent2.gene[key]
                child_gene[key] = value

        return Individual(child_gene)

    def crossover(self,parent1, parent2, crossover_prob=0.5):
        if random.random() < crossover_prob:
            return self.uniform_crossover(parent1, parent2)
        else:
            return self.blend_crossover(parent1, parent2)

    def mutate(self,individual, mutation_chance=0.1):
        mutated_gene = {}
        for key in individual.gene:
            if random.random() < self.mutation_chance:
                value_type = self.param_distributions[key][1]
                if value_type == int or value_type == float:
                    low, high = self.param_distributions[key][0]
                
                if value_type == bool:
                    mutated_gene[key] = not individual.gene[key]
                elif value_type == str:
                    mutated_gene[key] = random.choice(self.param_distributions[key][0])
                elif value_type == int:
                    mutated_gene[key] = random.randint(low,high)
                else:
                    mutated_gene[key] = random.uniform(low, high)
            else:
                mutated_gene[key] = individual.gene[key]

        return Individual(mutated_gene)
    
    def has_converged(self, patience=5):
        try:
            if len(self.best_scores) < patience + 1:
                return False
            return ((self.best_scores[-patience:] - self.best_scores[-1:]) < self.convergence_tolerance)
        except TypeError:
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
    
    def add_new_population(self):
        new_population_count = int(self.population*self.new_population_ratio)
        indis = self.indis
        for i in range(new_population_count):
            indi = self.get_new_indi(self.param_distributions)
            indis[i] = Individual(indi)
        self.indis = indis

    def search(self):
        self.spawn()
        for generation in range(self.max_generation):
            indis = self.indis
            for _ in range(self.population):
                parent1, parent2 = self.select()
                child = self.crossover(parent1,parent2)
                child = self.mutate(child)
                indis.append(child)
            self.indis = indis
            self.sort_indis()
            print(f"Generation {generation} : Best Score : {self.best_scores[-1]}, Avg Score : {self.average_scores[-1]}")
            if self.has_converged():
                break
            if random.random() < self.new_population_chance:
                self.add_new_population()
            
    def get_best_params(self):
        return self.indis[-1].gene
    
