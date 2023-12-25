from sklearn.model_selection import KFold, cross_val_score
from individual import Individual
from sklearn.base import clone
import random

class Optimizer:
    def __init__(self,X,y,model,ranges,population=10,n_splits = 5,selection_chance=0.5):
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.X = X
        self.y = y
        self.model = model
        self.ranges = ranges
        self.population = population
        self.selection_chance = selection_chance
        self.indis = []
        self.roulette_probs = []

        range_type = {}
        for key,value in ranges.items():

            range_type[key] = type(value[0])
        self.range_type = range_type
    
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
        range_type = self.range_type
        for _ in range(self.population):
            indi = []
            for key,value in ranges.items():
                if range_type[key] is int:
                    indi.append(random.randint(value[0],value[1]))
                else:
                    indi.append(random.uniform(value[0],value[1]))
            indis.append(indi)
        self.indis = indis
        self.sort_indis()

    def sort_indis(self):
        indis = self.indis
        for i in indis:
            i.score = self.getScore(i)
        indis.sort(key=lambda x: x.score)
        self.indis = indis
        sum_score = sum((i.score for i in indis))
        roulette_probs = [i.score/sum_score for i in indis]
        self.roulette_probs = roulette_probs
    
    def select(self):
        pass

    def crossover(self):
        pass

    def mutate(self):
        pass

    def hasConvergered(self):
        pass
