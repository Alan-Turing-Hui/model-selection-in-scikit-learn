from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from plot import ConfusionMatrixPlotter

class StratifiedKFoldEvaluator:
    def __init__(self, n_splits=5):
        """
        Initialize the Stratified K-Fold Evaluator.

        Parameters:
        n_splits (int): Number of folds for cross-validation.
        """
        self.n_splits = n_splits
        self.skf = StratifiedKFold(n_splits=self.n_splits)

    def evaluate(self, X, y, model, labels, metrics_save_path=None, cm_save_path=None):
        """
        Evaluate the given model using Stratified K-Fold cross-validation.

        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model: Scikit-learn model to evaluate.
        labels: True labels.

        Returns:
        None
        """
        total_confusion_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))))
        
        # Metrics storage
        precision_scores, recall_scores, f1_scores, accuracy_scores = [], [], [], []

        for train_index, test_index in self.skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            precision_scores.append(report['macro avg']['precision'])
            recall_scores.append(report['macro avg']['recall'])
            f1_scores.append(report['macro avg']['f1-score'])
            accuracy_scores.append(report['accuracy'])
            total_confusion_matrix += confusion_matrix(y_test, y_pred)

        # Plotting the metrics
        self._plot_metrics(precision_scores, recall_scores, f1_scores, accuracy_scores, model, metrics_save_path)
        self._plot_confusion_matrix(total_confusion_matrix, labels, model.__class__.__name__, cm_save_path)

    def _plot_metrics(self, precision_scores, recall_scores, f1_scores, accuracy_scores, model, save_path=None):
        """
        Plot the performance metrics.

        Parameters:
        precision_scores (list): List of precision scores for each fold.
        recall_scores (list): List of recall scores for each fold.
        f1_scores (list): List of F1 scores for each fold.
        accuracy_scores (list): List of accuracy scores for each fold.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))  # Create a new figure for metrics
        plt.plot(range(1, self.n_splits + 1), precision_scores, label='Precision')
        plt.plot(range(1, self.n_splits + 1), recall_scores, label='Recall')
        plt.plot(range(1, self.n_splits + 1), f1_scores, label='F1-score')
        plt.plot(range(1, self.n_splits + 1), accuracy_scores, label='Accuracy')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title(f'{model.__class__.__name__} Performance Metrics per Fold')
        plt.xticks(range(1, self.n_splits + 1))
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def _plot_confusion_matrix(self, cm, labels, model_name, save_path=None):
        """"
        Plot the confusion matrix.

        Parameters:

        Returns:
        None
        """
        plt.figure(figsize=(8, 6))  # Create a new figure for the confusion matrix
        cm_plotter = ConfusionMatrixPlotter(labels, cm, model_name)
        cm_plotter.plot(save_path)