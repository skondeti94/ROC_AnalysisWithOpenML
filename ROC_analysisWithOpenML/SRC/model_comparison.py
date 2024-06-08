from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, auc
import numpy as np

def compare_decision_trees(X, y, criterion):
    dtc = DecisionTreeClassifier(criterion=criterion)
    param_grid = {"min_samples_leaf": [2, 4, 6, 8, 10, 12]}
    
    grid_search = GridSearchCV(dtc, param_grid=param_grid, scoring='roc_auc', cv=10)
    grid_search.fit(X, y)
    
    best_dtc = grid_search.best_estimator_
    
    y_scores = cross_val_predict(best_dtc, X, y, method="predict_proba", cv=10)
    
    auc_scores = []

    for i in range(np.max(y) + 1):
        y_binary = (y == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
    
    return auc_scores, y_scores
