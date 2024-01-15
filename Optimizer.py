from sklearn.model_selection import KFold, cross_val_score
from individual import Individual
from sklearn.base import clone
import random
import bisect

class Optimizer:
    def __init__(self,X,y,model,ranges,max_generation=10,selection="both",population=10,n_splits = 5,selection_chance=0.5,tournament_split=2):
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.X = X
        self.y = y
        self.model = model
        self.max_generation = max_generation
        self.ranges = ranges
        self.selection = selection
        self.tournament_split = tournament_split
        self.population = population
        self.selection_chance = selection_chance
        self.indis = []
        self.roulette_probs = []
        self.best_scores = []
        self.average_scores = []
    
    def getScore(self,indi):
        new_model = clone(self.model)
        new_params = new_model.get_params()
        
        for param, value in indi.gene.items():
            new_params[param] = value
        
        new_model.set_params(**new_params)
        cv_scores = cross_val_score(new_model, self.X, self.y, cv=self.kfold, scoring='neg_mean_squared_error')
        return -(cv_scores.mean())

    def spawn(self):
        indis = []
        ranges = self.ranges
        for _ in range(self.population):
            indi = {}
            for key,value in ranges.items():
                value_type = ranges[key][1]
                # print("Hello")
                low, high = ranges[key][0][0], ranges[key][0][1]
                if value_type == bool:
                    indi[key] = random.choice((True,False))
                elif value_type == str:
                    indi[key] = random.choice(ranges[key][0])
                elif value[1] == int:
                    indi[key] = random.randint(low,high)
                else:
                    indi[key] = random.uniform(low,high)
            indis.append(Individual(indi))
        self.indis = indis
        self.sort_indis()

    def sort_indis(self):
        indis = self.indis
        for i in indis:
            if i.score == -1:
                i.score = self.getScore(i)
        indis.sort(key=lambda x: x.score)
        indis = indis[-self.population:]
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
            value_type = self.ranges[key][1]
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

    def mutate(self,individual, mutation_prob=0.1):
        mutated_gene = {}
        for key in individual.gene:
            if random.random() < mutation_prob:
                value_type = self.ranges[key][1]
                low, high = self.ranges[key][0]
                
                if value_type == bool:
                    mutated_gene[key] = not individual.gene[key]
                elif value_type == str:
                    mutated_gene[key] = random.choice(self.ranges[key][0])
                elif value_type == int:
                    mutated_gene[key] = random.randint(low,high)
                else:
                    mutated_gene[key] = random.uniform(low, high)
            else:
                mutated_gene[key] = individual.gene[key]

        return Individual(mutated_gene)
    
    def has_converged(self, tolerance=1e-5, patience=5):
        try:
            if len(self.best_scores) < patience + 1:
                return False
            return ((self.best_scores[-patience:] - self.best_scores[-1:]) < tolerance)
        except TypeError:
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
    
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
            
    def get_best_params(self):
        return self.indis[-1].gene