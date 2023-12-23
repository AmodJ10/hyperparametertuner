from sklearn.linear_model import ElasticNet
import numpy as np

class Individual:
    def __init__(self):
        self.alpha = np.random.uniform(0.1, 1.0),
        self.l1_ratio = np.random.uniform(0.0, 1.0),
        self.fit_intercept = np.random.choice([True, False]),
        self.precompute = np.random.choice([True, False]),
        self.max_iter = np.random.randint(100, 1000),
        self.copy_x = np.random.choice([True, False]),
        self.tol = np.random.uniform(1e-5, 1e-2),
        self.warm_start = np.random.choice([True, False]),
        self.positive = np.random.choice([True, False]),
        self.random_state = np.random.randint(1, 100),
        self.selection =  np.random.choice(['cyclic', 'random'])
        self.fitness = None  

    def evaluate(self, X_train, y_train, X_val, y_val):
        
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            precompute=self.precompute,
            max_iter=self.max_iter,
            copy_X=self.copy_x,
            tol=self.tol,
            warm_start=self.warm_start,
            positive=self.positive,
            random_state=self.random_state,
            selection=self.selection
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        
        mse = np.mean((predictions - y_val) ** 2)
        self.fitness = -mse 

    def __repr__(self):
        return (f"Individual(alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
                f"fit_intercept={self.fit_intercept}, precompute={self.precompute}, "
                f"max_iter={self.max_iter}, copy_x={self.copy_x}, tol={self.tol}, "
                f"warm_start={self.warm_start}, positive={self.positive}, "
                f"random_state={self.random_state}, selection={self.selection})")

def generate_random_hyperparameters():
    return {
        'alpha': np.random.uniform(0.1, 1.0),
        'l1_ratio': np.random.uniform(0.0, 1.0),
        'fit_intercept': np.random.choice([True, False]),
        'precompute': np.random.choice([True, False]),
        'max_iter': np.random.randint(100, 1000),
        'copy_x': np.random.choice([True, False]),
        'tol': np.random.uniform(1e-5, 1e-2),
        'warm_start': np.random.choice([True, False]),
        'positive': np.random.choice([True, False]),
        'random_state': np.random.randint(1, 100),
        'selection': np.random.choice(['cyclic', 'random'])
    }


