# Utilization of sklearn.model_selection API
- Tools for model selection, such as cross validation and hyper-parameter tuning.

## Stratified K-Fold cross-validator
- **Description:**
    - Stratified K-Fold cross-validation ensures that each fold of the dataset maintains the same proportion of class labels as the entire dataset. This is especially useful for imbalanced datasets, as it helps in producing more reliable and consistent model evaluations by ensuring that each class is adequately represented in every fold. 

1. **StratifiedKFold python file**
- **Description:**
    - Define a class StratifiedKFoldEvaluator which contains:
        - `evaluate()` function,
        - `_plot_metrics` function,
        - `_plot_confusion_matrix` function. 
