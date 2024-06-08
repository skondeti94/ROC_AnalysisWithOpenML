#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install openml


# In[ ]:


import openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
 
 
def load_dataset(dataset_id):
    dataset_grab = openml.datasets.get_dataset(dataset_id)
    P, y, _, _ = dataset_grab.get_data(target=dataset_grab.default_target_attribute)
    encoder_label = LabelEncoder()
    y = encoder_label.fit_transform(y)
    if isinstance(P, pd.DataFrame):
        for column in P.columns:
            if P[column].dtype == 'object':
                P[column] = LabelEncoder().fit_transform(P[column])
            elif P[column].dtype == 'category':
                P[column] = P[column].cat.codes
 
    return P, y
 
 
def compare_decision_trees(dataset_id, criterion):
    P, y = load_dataset(dataset_id)
 
 
    dtc1 = DecisionTreeClassifier(criterion=criterion)
 
   
    my_parameters = {"min_samples_leaf": [2, 4, 6, 8, 10, 12]}
 
 
    grid_searching = GridSearchCV(dtc1, param_grid=my_parameters, scoring='roc_auc', cv=10)
    grid_searching.fit(P, y)
 
 
    best_dtc = grid_searching.best_estimator_
 
 
    y_scores = cross_val_predict(best_dtc, P, y, method="predict_proba", cv=10)
    
    
    auc_score_toprint = []
 
    plt.figure()  
    for j in range(np.max(y) + 1):
        y_binary = (y == j).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_scores[:, j])
        roc_auc = auc(fpr, tpr)
       
        auc_score_toprint.append(roc_auc)
       
   
        plt.plot(fpr, tpr, label=f"Class {j} (AUC = {roc_auc:.2f})")
 
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curves for Decision Tree ({criterion}) on Dataset {dataset_id}")
    plt.legend()
    plt.show()
   
    return auc_score_toprint
 
dataset_id_1 = 4137
dataset_id_2 = 4134
 
 
 
auc_scores_entropy_dset1 = compare_decision_trees(dataset_id_1, "entropy")
print("AUC Scores of Entropy(Dataset-1):", auc_scores_entropy_dset1)
 
auc_scores_gini_dset1 = compare_decision_trees(dataset_id_1, "gini")
print("AUC Scores of Gini(Dataset-1):", auc_scores_gini_dset1)
 
auc_scores_entropy_dset2 = compare_decision_trees(dataset_id_2, "entropy")
print("AUC Scores  of Entropy(Dataset-2):", auc_scores_entropy_dset2)
 
auc_scores_gini_dset2 = compare_decision_trees(dataset_id_2, "gini")
print("AUC Scores of Gini(Dataset-2):", auc_scores_gini_dset2)


# In[ ]:




