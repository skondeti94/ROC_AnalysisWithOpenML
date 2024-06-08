import openml
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    if isinstance(X, pd.DataFrame):
        for column in X.columns:
            if X[column].dtype == 'object':
                X[column] = LabelEncoder().fit_transform(X[column])
            elif X[column].dtype.name == 'category':
                X[column] = X[column].cat.codes

    return X, y
