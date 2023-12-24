from sklearn.model_selection import KFold, cross_val_score
from individual import Individual
from sklearn.base import clone

class Optimizer:
    def __init__(self,X,y,model,ranges,population=10,n_splits = 4):
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.X = X
        self.y = y
        pass
    
    def getScore(self,indi):
        new_model = clone(self.model)
        new_params = new_model.get_params()
        
        for param, value in indi.params.items():
            if param in new_params:
                new_params[param] = value
        
        new_model.set_params(**new_params)
        cv_scores = cross_val_score(new_model, self.X, self.y, cv=self.kfold, scoring='accuracy')
        return cv_scores.mean()

    def spawn(self):
        pass

    def select(self):
        pass

    def crossover(self):
        pass

    def mutate(self):
        pass

    def hasConvergered(self):
        pass
