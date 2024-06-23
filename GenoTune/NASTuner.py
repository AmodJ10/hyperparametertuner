from sklearn.model_selection import KFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Flatten
import bisect 
import random
from joblib import Parallel, delayed

class Individual:
    def __init__(self,gene):
        self.gene = gene
        self.score = -1
        self.model = None

class BaseTuner:
    def __init__(
            self,
            X,
            y,
            input_nodes,
            output_nodes,
            output_activation,
            optimizers,
            hidden_activations,
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
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.output_activation = output_activation
        self.optimizers = optimizers
        self.hidden_activations = hidden_activations
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

    def checkSize(self,gene):
        num_layers = gene[0]
        num_params = 0
        for i in range(2,num_layers):
            num_params += gene[i]*gene[i+1]

        if num_params>self.max_params:
            return False
        return True

    def crossValScore(model,x,y,cv):
        pass

    def getScore(self,indi):
        if indi.score != -1:
            return indi.score
        new_model = self.build_model(indi.gene)
        cv_scores= cross_val_score(new_model, self.X, self.y, cv=self.kfold) # Change it for tensorflow models
        return cv_scores.mean()

    def build_model(self,gene):
        model = Sequential()
        num_layers = gene[0]
        optimizer = gene[1]
        for i in range(2,num_layers+1):
            model.add(Dense(gene[i][0],activation=gene[i][1]))
        model.compile(optimizer=optimizer,loss=self.loss)
        return model
    
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

    def get_new_indi(self):
        num_layers = random.randint(self.min_layers,self.max_layers) # TODO define min_layers, max_layers
        gene = [[self.input_nodes,"relu"], [self.output_nodes, self.output_activation]]
        curr_layers = 2
        count = 0
        while curr_layers<num_layers and count<5:
            curr_pos = random.randint(2,curr_layers-1)
            curr_nodes = random.randint(self.min_nodes, self.max_nodes) # TODO define min_nodes, max_nodes
            curr_activation = random.choice(self.hidden_activations)
            if gene[curr_pos-1][0]*curr_nodes + gene[curr_pos+1][0]*curr_nodes - gene[curr_pos-1][0]*gene[curr_pos+1][0]<self.max_params:
                gene.insert(curr_pos,[curr_nodes,curr_activation])
                curr_layers+=1
                count=0
            else:
                count+=1
        optimizer = random.choice(self.optimizers)
        gene = [num_layers, optimizer] + gene
        return gene

    def spawn(self):
        indis = []
        # TODO figure out min_layers, max_layers, min_nodes, max_nodes and update to self.
        for _ in range(self.population):
            indi = self.get_new_indi()
            indis.append(Individual(indi))
        self.indis = indis
        self.sort_indis()

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
            indi = self.get_new_indi()
            indis[i] = Individual(indi)
        self.indis = indis

    def crossover(self,parent1,parent2):
        pass

    def mutate(self,child):
        pass

    def search(self):
        self.spawn()
        for generation in range(self.max_generation):
            indis = self.indis
            for _ in range(self.population):
                parent1, parent2 = self.select()
                child = self.crossover(parent1,parent2) # TODO define crossover
                child = self.mutate(child) # TODO define mutate
                indis.append(child)
            self.indis = indis
            self.sort_indis()
            print(f"Generation {generation} : Best Score : {self.best_scores[-1]}, Avg Score : {self.average_scores[-1]}")
            if self.has_converged():
                break
            if random.random() < self.new_population_chance:
                self.add_new_population()


    

        