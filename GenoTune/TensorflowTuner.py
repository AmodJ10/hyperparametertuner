from sklearn.model_selection import KFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Flatten

class Individual:
    def __init__(self,gene):
        self.gene = gene
        self.score = -1
        self.model = None

class TensorflowTuner:
    def __init__(
            self,
            X,
            y,
            input_shape,
            output_layer,
            loss,
            epochs,
            batch_size,
            max_params,
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
        self.input_shape = input_shape
        self.output_layer = output_layer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_params = max_params
        self.max_generation = max_generation
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

    def controlSize(self,gene):
        num_layers = gene[0]
        num_params = 0
        for i in range(2,num_layers):
            num_params += gene[i]*gene[i+1]

        if num_params>self.max_params:
            return False
        return True

    def build_model(self,gene):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        num_layers = gene[0]
        optimizer = gene[1]
        for i in range(2,num_layers+1):
            model.add(Dense(gene[i][0],activation=gene[i][1]))
        model.add(Dense(self.output_layer[0],activation=self.output_layer[1]))
        model.compile(optimizer=optimizer,loss=self.loss)
        return model

    def getScore(self,indi):
        if indi.score != -1:
            return indi.score
        new_model = self.build_model(indi.gene)
        cv_scores= cross_val_score(new_model, self.X, self.y, cv=self.kfold)
        return cv_scores.mean()

        