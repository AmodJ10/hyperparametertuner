from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

def get_kfold_accuracy(model, X, y, n_splits=5):
    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform K-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    return cv_scores.mean()  # Return the mean accuracy score across all folds

# Load sample data
iris = load_iris()
X = iris.data
y = iris.target

# Initialize the model
model = LogisticRegression(max_iter=1000)  # Example model, you can use your own

# Get K-fold cross-validation accuracy
accuracy = get_kfold_accuracy(model, X, y)
print(f'K-fold Cross-Validation Accuracy: {accuracy:.4f}')
