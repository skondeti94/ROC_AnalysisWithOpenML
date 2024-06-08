import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_curves(y, y_scores, title):
    plt.figure()
    for i in range(np.max(y) + 1):
        y_binary = (y == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend()
    plt.show()
