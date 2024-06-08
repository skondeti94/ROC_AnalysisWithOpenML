from data_loading import load_dataset
from model_comparison import compare_decision_trees
from visualization import plot_roc_curves

def main():
    dataset_id_1 = 4137
    dataset_id_2 = 4134
    
    X1, y1 = load_dataset(dataset_id_1)
    X2, y2 = load_dataset(dataset_id_2)
    
    auc_scores_entropy_dset1, y_scores_entropy_dset1 = compare_decision_trees(X1, y1, "entropy")
    print("AUC Scores for Entropy (Dataset 1):", auc_scores_entropy_dset1)
    plot_roc_curves(y1, y_scores_entropy_dset1, f"ROC Curves for Decision Tree (Entropy) on Dataset {dataset_id_1}")
    
    auc_scores_gini_dset1, y_scores_gini_dset1 = compare_decision_trees(X1, y1, "gini")
    print("AUC Scores for Gini (Dataset 1):", auc_scores_gini_dset1)
    plot_roc_curves(y1, y_scores_gini_dset1, f"ROC Curves for Decision Tree (Gini) on Dataset {dataset_id_1}")
    
    auc_scores_entropy_dset2, y_scores_entropy_dset2 = compare_decision_trees(X2, y2, "entropy")
    print("AUC Scores for Entropy (Dataset 2):", auc_scores_entropy_dset2)
    plot_roc_curves(y2, y_scores_entropy_dset2, f"ROC Curves for Decision Tree (Entropy) on Dataset {dataset_id_2}")
    
    auc_scores_gini_dset2, y_scores_gini_dset2 = compare_decision_trees(X2, y2, "gini")
    print("AUC Scores for Gini (Dataset 2):", auc_scores_gini_dset2)
    plot_roc_curves(y2, y_scores_gini_dset2, f"ROC Curves for Decision Tree (Gini) on Dataset {dataset_id_2}")

if __name__ == "__main__":
    main()
