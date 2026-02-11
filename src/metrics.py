import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def print_metrics(metrics):
    """Pretty print evaluation metrics"""
    print(f"\nModel Performance:")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1']:.3f}")
    print("\nClassification Report:")
    print(metrics['report'])

def get_arnold_predictions(X_test, y_test, y_pred, lfw_images, h, w):
    """Get indices of actual Arnold images in test set"""
    arnold_test_idx = np.where(y_test == 1)[0]
    return arnold_test_idx, y_pred[arnold_test_idx]
